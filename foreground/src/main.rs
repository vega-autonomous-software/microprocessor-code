use std::sync::mpsc;
use std::thread;
use std::time::Duration;

mod pythonspawn;
use pythonspawn::runpythonfile_stream;

fn main() {
    let (tx, rx) = mpsc::channel::<(&'static str, String)>();

    let tx_engine = tx.clone();
    let handle_engine = thread::spawn(move || {
        runpythonfile_stream("python/engine.py", "engine.py", tx_engine);
    });

    // Give FSDS time to start
    thread::sleep(Duration::from_secs(25));

    let tx_control = tx.clone();
    let handle_control = thread::spawn(move || {
        runpythonfile_stream(
            "python/control_input_node.py",
            "control_input_node.py",
            tx_control,
        );
    });

    let tx_imu = tx.clone();
    let handle_imu = thread::spawn(move || {
        runpythonfile_stream(
            "python/imu_speed_node.py",
            "imu_speed_node.py",
            tx_imu,
        );
    });

    let tx_vision = tx.clone();
    let handle_vision = thread::spawn(move || {
        runpythonfile_stream(
            "python/vision_node.py",
            "vision_node.py",
            tx_vision,
        );
    });

    drop(tx);

    while let Ok((tag, line)) = rx.recv() {
        println!("[{tag}] {line}");
    }

    handle_engine.join().expect("engine.py thread panicked");
    handle_control
        .join()
        .expect("control_input_node.py thread panicked");
    handle_imu
        .join()
        .expect("imu_speed_node.py thread panicked");
    handle_vision
        .join()
        .expect("vision_node.py thread panicked");
}