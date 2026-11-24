//! BAM reader/writer stubs

use anyhow::Result;

pub fn read_bam(path: &str) -> Result<()> {
    let _content = std::fs::read(path)?;
    Ok(())
}

pub fn write_bam(path: &str, _content: &[u8]) -> Result<()> {
    std::fs::write(path, _content)?;
    Ok(())
}
