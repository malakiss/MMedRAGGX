"""Domain models for radiology images and reports."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RadiologyImage:
    """Represents a radiology image."""

    image_id: str
    image_path: str
    image_root: str
    report: str
    modality: str = "chest_xray"
    findings: Optional[list[str]] = None  # List of extracted medical findings
    image_embedding: Optional[list[float]] = None  # Visual embedding from OpenCLIP
    report_embedding: Optional[list[float]] = None  # Text embedding from report

    def __post_init__(self):
        """Initialize findings if None."""
        if self.findings is None:
            self.findings = []

    def get_full_path(self) -> str:
        """Get full image path by combining root and relative path."""
        from pathlib import Path
        if self.image_root:
            return str(Path(self.image_root) / self.image_path)
        return self.image_path
    #==================================================================================
#--TODO: Implement a more sophisticated findings extraction method

    def extract_findings(self) -> list[str]:
        """Extract medical findings from report using keyword matching."""
        if not self.report:
            return []
        
        finding_keywords = [
            "opacity", "infiltrate", "consolidation", "effusion",
            "pneumothorax", "cardiomegaly", "atelectasis", "nodule",
            "pneumonia", "edema", "mass", "fracture", "congestion"
        ]
        
        findings = []
        report_lower = self.report.lower()
        for keyword in finding_keywords:
            if keyword in report_lower:
                findings.append(keyword)
        
        self.findings = findings
        return findings

#====================================================================================================
@dataclass
class RadiologyReport:
    """Represents a radiology report."""

    report_id: str
    image_id: str
    text: str
    findings: Optional[list[str]] = None

    def __post_init__(self):
        """Initialize findings if None."""
        if self.findings is None:
            self.findings = []


@dataclass
class RadiologyDatasetMetadata:
    """Metadata for the radiology dataset."""

    collection_date: str
    total_images: int
    dataset_name: str = "IU X-Ray"
    modality: str = "chest_xray"


@dataclass
class RadiologyDataset:
    """Complete radiology dataset with metadata."""

    metadata: RadiologyDatasetMetadata
    images: list[RadiologyImage]
