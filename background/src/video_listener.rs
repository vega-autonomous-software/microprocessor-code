use std::fs::File;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

pub fn start_video_listener(running: Arc<AtomicBool>) {
    let listener =
        TcpListener::bind("0.0.0.0:84").expect("Failed to bind port 84");

    println!("Video listener running on port 84");

    listener
        .set_nonblocking(true)
        .expect("Cannot set non-blocking");

    while running.load(Ordering::SeqCst) {
        if let Ok((mut stream, _)) = listener.accept() {
            let mut file =
                File::create("video.ts").expect("Cannot create video.ts");

            let mut buffer = [0u8; 8192];

            while running.load(Ordering::SeqCst) {
                match stream.read(&mut buffer) {
                    Ok(0) => break,
                    Ok(n) => {
                        file.write_all(&buffer[..n]).unwrap();
                    }
                    Err(_) => break,
                }
            }
        }
    }

    println!("Video listener stopped.");
}
