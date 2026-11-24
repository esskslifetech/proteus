//! Parquet reader/writer

use anyhow::Result;

pub fn read_parquet(path: &str) -> Result<()> {
    let _file = std::fs::File::open(path)?;
    Ok(())
}

pub fn write_parquet(path: &str, _content: &[u8]) -> Result<()> {
    std::fs::write(path, _content)?;
    Ok(())
}