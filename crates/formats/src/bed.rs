//! BED (Browser Extensible Data) reader/writer

use anyhow::Result;

pub fn read_bed(path: &str) -> Result<()> {
    let _content = std::fs::read_to_string(path)?;
    Ok(())
}

pub fn write_bed(path: &str, _content: &str) -> Result<()> {
    std::fs::write(path, _content)?;
    Ok(())
}