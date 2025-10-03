import os
import re
import csv
import glob
import pandas as pd
import tensorflow as tf
from datetime import datetime
from nvflare.fuel.flare_api.flare_api import new_secure_session, Session
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ---------- Utilities ----------
ANSI_ESCAPE_RE = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')

def strip_ansi(line):
    return ANSI_ESCAPE_RE.sub('', line)

def decode_tensor_to_str(tensor_proto):
    tensor = tf.make_ndarray(tensor_proto)
    if tensor.size == 1:
        val = tensor.item()
        return val.decode("utf-8") if isinstance(val, bytes) else str(val)
    return ",".join(s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in tensor.flatten())

# ---------- TensorBoard Data Loaders ----------
def load_scalar_data(event_acc, tags):
    return {
        tag: pd.DataFrame([{'step': e.step, 'wall_time': e.wall_time, 'value': e.value} for e in event_acc.Scalars(tag)])
        for tag in tags
    }

def load_text_data(event_acc, tags):
    return {
        tag: pd.DataFrame([{'step': e.step, 'wall_time': e.wall_time, 'value': e.text} for e in event_acc.Text(tag)])
        for tag in tags
    }

def load_tensor_text_data(event_acc, tags):
    text_data = {}
    for tag in tags:
        tensor_events = event_acc.Tensors(tag)
        rows = [{"step": e.step, "wall_time": e.wall_time, "value": decode_tensor_to_str(e.tensor_proto)} for e in tensor_events]
        clean_tag = tag.split("/")[0]
        text_data[clean_tag] = pd.DataFrame(rows)
    return text_data

def get_all_tensorflow_event_data(log_dir):
    if not os.path.isdir(log_dir):
        print(f"Error: Log directory not found at '{log_dir}'")
        return {}

    event_acc = EventAccumulator(log_dir, size_guidance={k: 0 for k in ["scalars", "histograms", "images", "tensors", "text"]})
    event_acc.Reload()
    tags = event_acc.Tags()

    print(f"Available tags: {tags}")
    scalar_data = load_scalar_data(event_acc, tags.get('scalars', []))
    text_data = load_text_data(event_acc, tags.get('text', []))
    tensor_text_data = load_tensor_text_data(event_acc, [t for t in tags.get('tensors', []) if 'text_summary' in t])
    text_data.update(tensor_text_data)

    data = {}
    if scalar_data:
        data['scalars'] = scalar_data
    if text_data:
        data['text'] = text_data

    return data

# ---------- CSV Writer ----------
def save_local_metrics_to_csv(scalar_data, text_data, filename="val_local_metrics.csv", path=None):
    scalar_tags = ["val_local_f1-score", "val_local_precision", "val_local_recall", "val_local_accuracy", "train_epoch_runtime"]
    text_tags = ["epoch_start_timestamp", "epoch_end_timestamp"]

    missing_scalars = [tag for tag in scalar_tags if tag not in scalar_data]
    missing_texts = [tag for tag in text_tags if tag not in text_data]
    if missing_scalars or missing_texts:
        print(f"Warning: Missing tags - Scalars: {missing_scalars}, Text: {missing_texts}")
        return

    def prepare_df(df, col):
        return df[["step", "value"]].rename(columns={"value": col, "step": "epoch"})

    df = prepare_df(scalar_data[scalar_tags[0]], scalar_tags[0])
    for tag in scalar_tags[1:]:
        df = df.merge(prepare_df(scalar_data[tag], tag), on="epoch")

    for tag in text_tags:
        temp = text_data[tag][["step", "value"]].rename(columns={"value": tag})
        temp["epoch"] = temp["step"] // 2
        df = df.merge(temp[["epoch", tag]], on="epoch")

    df.insert(0, "round", df["epoch"] // 10 + 1)

    if path:
        os.makedirs(path, exist_ok=True)
        filename = os.path.join(path, filename)

    df.to_csv(filename, index=False)
    print(f"Saved local validation metrics to {filename}")

# ---------- Round Timing Extractor ----------
def extract_round_timings(base_dir, system_name, job_id, output_dir):
    log_pattern = os.path.join(base_dir, system_name, "logs", "task_0", "frontier*.log")
    print(f"[DEBUG] Looking for logs at {log_pattern}")
    log_files = glob.glob(log_pattern)

    if not log_files:
        raise FileNotFoundError(f"No log file matching pattern {log_pattern}")
    server_log_path = log_files[0]
    print(f"[DEBUG] Using log file: {server_log_path}")

    start_pattern = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+).*?Round (\d+) started')
    end_pattern = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+).*?Round (\d+) finished')

    round_start_times, round_end_times = {}, {}

    with open(server_log_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            clean_line = strip_ansi(line.strip())
            if match := start_pattern.match(clean_line):
                timestamp, round_num = match.groups()
                round_start_times[int(round_num)] = timestamp
            elif match := end_pattern.match(clean_line):
                timestamp, round_num = match.groups()
                round_end_times[int(round_num)] = timestamp

    output_csv = os.path.join(output_dir, "Communication_round_timestamp.csv")
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['round', 'round_start_timestamp', 'round_end_timestamp', 'round_duration'])

        all_rounds = sorted(set(round_start_times) | set(round_end_times))
        for r in all_rounds:
            start_str = round_start_times.get(r)
            end_str = round_end_times.get(r)
            duration_str = ''
            if start_str and end_str:
                try:
                    fmt = "%Y-%m-%d %H:%M:%S,%f"
                    duration = (datetime.strptime(end_str, fmt) - datetime.strptime(start_str, fmt)).total_seconds()
                    duration_str = f"{duration:.3f}"
                except Exception as e:
                    duration_str = f"Error: {e}"
            writer.writerow([r, start_str or '', end_str or '', duration_str])

    print(f"✓ Saved round timing CSV to {output_csv}")

# ---------- Job Callback ----------
def sample_cb(session: Session, job_id: str, job_meta, *cb_args, **cb_kwargs):
    if job_meta["status"] == "RUNNING":
        print(job_meta if cb_kwargs["cb_run_counter"]["count"] < 3 else ".", end="")
    else:
        print("\n" + str(job_meta))
    cb_kwargs["cb_run_counter"]["count"] += 1
    return True

# ---------- Main Execution ----------
if __name__ == "__main__":
    location = os.environ.get("LOCATION")
    job_name = os.environ.get("JOB")
    username = "admin@ornl.gov"
    admin_user_dir = os.path.join(location, "example_intranode", username)
    job_path = os.path.join(admin_user_dir, "transfer", job_name)

    sess = new_secure_session(username=username, startup_kit_location=admin_user_dir)
    job_id1 = sess.submit_job(job_path)
    print(f"{job_id1} was submitted")

    sess.monitor_job(job_id1, cb=sample_cb, cb_run_counter={"count": 0})
    sess.download_job_result(job_id1)

    sess.shutdown('client')
    sess.shutdown('server')
