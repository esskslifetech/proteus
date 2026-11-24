//! GFF reader/writer stubs

use anyhow::Result;

pub fn read_gff(path: &str) -> Result<()> {
    let _content = std::fs::read_to_string(path)?;
    Ok(())
}

pub fn write_gff(path: &str, _content: &str) -> Result<()> {
    std::fs::write(path, _content)?;
    Ok(())
}
