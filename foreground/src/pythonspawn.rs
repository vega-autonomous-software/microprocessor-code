// src/pythonspawn.rs
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::sync::mpsc::Sender;

/// Spawns a Python script, sends one line to its stdin, and streams stdout lines back to Rust via mpsc.
/// Each stdout line is sent as (script_tag, line).
pub fn runpythonfile_stream(
    inputpath: &str,
    script_tag: &'static str,
    tx: Sender<(&'static str, String)>,
) {
    let mut child = Command::new("python")
        .arg(inputpath)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap_or_else(|e| panic!("Failed to start Python script {inputpath}: {e}"));

    // Send initial input so Python's input() unblocks
    if let Some(stdin) = child.stdin.as_mut() {
        stdin
            .write_all(b"start\n")
            .unwrap_or_else(|e| panic!("Failed to write to Python stdin for {inputpath}: {e}"));
        // Optional but nice: flush immediately
        stdin
            .flush()
            .unwrap_or_else(|e| panic!("Failed to flush Python stdin for {inputpath}: {e}"));
    }
    // Close stdin so Python won't keep waiting for more input (unless your Python expects more)
    drop(child.stdin.take());

    // Stream stdout line-by-line
    let stdout = child
        .stdout
        .take()
        .unwrap_or_else(|| panic!("Failed to capture stdout for {inputpath}"));

    let reader = BufReader::new(stdout);

    for line_result in reader.lines() {
        match line_result {
            Ok(line) => {
                // Send each line to main
                if tx.send((script_tag, line)).is_err() {
                    // Receiver dropped; stop
                    break;
                }
            }
            Err(e) => {
                let _ = tx.send((script_tag, format!("[stdout read error] {e}")));
                break;
            }
        }
    }

    // Ensure the child exits
    let status = child
        .wait()
        .unwrap_or_else(|e| panic!("Failed to wait on Python script {inputpath}: {e}"));

    let _ = tx.send((
        script_tag,
        format!("[process exited] code={}", status.code().unwrap_or(-1)),
    ));
}