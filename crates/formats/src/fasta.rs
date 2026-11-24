//! FASTA reader/writer stubs

use anyhow::Result;

pub fn read_fasta(path: &str) -> Result<()> {
    let _content = std::fs::read_to_string(path)?;
    Ok(())
}

pub fn write_fasta(path: &str, _content: &str) -> Result<()> {
    std::fs::write(path, _content)?;
    Ok(())
}
