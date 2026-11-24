//! VCF reader/writer stubs

use anyhow::Result;

pub fn read_vcf(path: &str) -> Result<()> {
    let _content = std::fs::read_to_string(path)?;
    Ok(())
}

pub fn write_vcf(path: &str, _content: &str) -> Result<()> {
    std::fs::write(path, _content)?;
    Ok(())
}
