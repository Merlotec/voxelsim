use crossbeam_channel::{Receiver, Sender};
use js_sys::{ArrayBuffer, Uint8Array};
use serde::de::DeserializeOwned;
use wasm_bindgen::{prelude::*, JsCast};
use web_sys::{BinaryType, ErrorEvent, MessageEvent, WebSocket};

pub struct NetworkSubscriber<T> {
    sender: Sender<T>,
    ws_url: String,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> NetworkSubscriber<T> {
    pub fn new(ws_url: String) -> (Self, Receiver<T>) {
        let (sender, receiver) = crossbeam_channel::unbounded::<T>();

        (
            Self {
                sender,
                ws_url,
                _phantom: std::marker::PhantomData,
            },
            receiver,
        )
    }
}

impl<T: 'static + DeserializeOwned + Send> NetworkSubscriber<T> {
    pub fn start(&self) {
        web_sys::console::log_1(&format!("Starting WebSocket connection to {}...", self.ws_url).into());

        let ws = match WebSocket::new(&self.ws_url) {
            Ok(ws) => ws,
            Err(e) => {
                web_sys::console::error_1(&format!("Failed to create WebSocket: {:?}", e).into());
                return;
            }
        };

        // Set binary type to arraybuffer
        ws.set_binary_type(BinaryType::Arraybuffer);

        let sender = self.sender.clone();

        // onmessage callback
        let onmessage_callback = Closure::wrap(Box::new(move |e: MessageEvent| {
            if let Ok(arraybuffer) = e.data().dyn_into::<ArrayBuffer>() {
                let array = Uint8Array::new(&arraybuffer);
                let bytes = array.to_vec();

                // Parse: [4-byte length][bincode payload]
                if bytes.len() < 4 {
                    web_sys::console::error_1(
                        &format!("Received message too short: {} bytes", bytes.len()).into(),
                    );
                    return;
                }

                let msg_len = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;

                if bytes.len() < 4 + msg_len {
                    web_sys::console::error_1(
                        &format!(
                            "Incomplete message: expected {} bytes, got {}",
                            4 + msg_len,
                            bytes.len()
                        )
                        .into(),
                    );
                    return;
                }

                let payload = &bytes[4..4 + msg_len];

                // Deserialize with bincode
                match bincode::serde::decode_from_slice::<T, _>(payload, bincode::config::standard())
                {
                    Ok((data, _)) => {
                        if let Err(e) = sender.send(data) {
                            web_sys::console::error_1(&format!("Failed to send data: {}", e).into());
                        }
                    }
                    Err(e) => {
                        web_sys::console::error_1(
                            &format!("Failed to deserialize message: {}", e).into(),
                        );
                    }
                }
            } else {
                web_sys::console::warn_1(&"Received non-binary message".into());
            }
        }) as Box<dyn FnMut(MessageEvent)>);

        ws.set_onmessage(Some(onmessage_callback.as_ref().unchecked_ref()));
        onmessage_callback.forget(); // Keep closure alive

        // onopen callback
        let ws_url = self.ws_url.clone();
        let onopen_callback = Closure::wrap(Box::new(move |_| {
            web_sys::console::log_1(&format!("✓ WebSocket connected: {}", ws_url).into());
        }) as Box<dyn FnMut(JsValue)>);

        ws.set_onopen(Some(onopen_callback.as_ref().unchecked_ref()));
        onopen_callback.forget();

        // onerror callback
        let ws_url_err = self.ws_url.clone();
        let onerror_callback = Closure::wrap(Box::new(move |e: ErrorEvent| {
            web_sys::console::error_1(
                &format!("✗ WebSocket error on {}: {}", ws_url_err, e.message()).into(),
            );
        }) as Box<dyn FnMut(ErrorEvent)>);

        ws.set_onerror(Some(onerror_callback.as_ref().unchecked_ref()));
        onerror_callback.forget();

        // onclose callback
        let ws_url_close = self.ws_url.clone();
        let onclose_callback = Closure::wrap(Box::new(move |_| {
            web_sys::console::log_1(&format!("WebSocket closed: {}", ws_url_close).into());
        }) as Box<dyn FnMut(JsValue)>);

        ws.set_onclose(Some(onclose_callback.as_ref().unchecked_ref()));
        onclose_callback.forget();

        // Keep WebSocket alive by leaking it
        // In production, you'd store it in a Bevy Resource for proper lifetime management
        Box::leak(Box::new(ws));
    }
}
