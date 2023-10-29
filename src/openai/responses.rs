use actix_web::error;
use derive_more::{Display, Error};

#[derive(Debug, Display, Error)]
#[display(fmt = "Error: {}", data)]
pub struct APIError {
    data: String,
}

impl error::ResponseError for APIError {}

impl APIError {
    pub fn new(data: String) -> Self {
        Self { data }
    }
}
