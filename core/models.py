from typing import Dict, List, Optional

try:
    from pydantic import BaseModel
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = None

if PYDANTIC_AVAILABLE:
    class DBColumnModel(BaseModel):
        name: str
        type: str
        pk: bool = False
        nullable: bool = True
        unique: bool = False
        default: Optional[str] = None
        notes: Optional[str] = None

    class DBRelationshipModel(BaseModel):
        type: str
        to_table: str
        fk: Optional[str] = None
        ref: Optional[str] = None
        notes: Optional[str] = None

    class DBTableModel(BaseModel):
        name: str
        purpose: str
        columns: List[DBColumnModel] = []
        indexes: List[str] = []
        relationships: List[DBRelationshipModel] = []

    class DatabaseSchemaModel(BaseModel):
        tables: List[DBTableModel] = []
        assumptions: List[str] = []

    class FeatureModel(BaseModel):
        id: str
        name: str
        description: str
        priority: str
        user_stories: List[str] = []
        acceptance_criteria: List[str] = []

    class TechnicalRequirementsModel(BaseModel):
        platform: List[str] = []
        technologies: List[str] = []
        database: Optional[str] = None

    class RequirementsOutputModel(BaseModel):
        project_name: str
        features: List[FeatureModel] = []
        technical_requirements: TechnicalRequirementsModel
        user_roles: List[str] = []
        database_schema: Optional[DatabaseSchemaModel] = None

    class ColorSchemeModel(BaseModel):
        primary: str
        secondary: str
        accent: str
        surface: str
        background: str
        text_primary: str
        error: str
        success: str
        warning: str

    class TypographySizesModel(BaseModel):
        small: Optional[str] = None
        medium: Optional[str] = None
        large: Optional[str] = None
        xlarge: Optional[str] = None

    class TypographyWeightsModel(BaseModel):
        normal: Optional[str] = None
        bold: Optional[str] = None

    class TypographyModel(BaseModel):
        font_family: Optional[str] = None
        heading_font: Optional[str] = None
        body_font: Optional[str] = None
        sizes: Optional[TypographySizesModel] = None
        weights: Optional[TypographyWeightsModel] = None

    class NavigationModel(BaseModel):
        type: str
        items: List[str] = []

    class ComponentModel(BaseModel):
        name: str
        type: str
        position: Optional[str] = None
        size: Optional[str] = None
        styling: Optional[str] = None
        interaction: Optional[str] = None
        icon: Optional[str] = None
        has_image: bool = False
        image_type: Optional[str] = None

    class ScreenModel(BaseModel):
        name: str
        purpose: str
        key_components: List[ComponentModel] = []
        user_flow: Optional[str] = None

    class DesignOutputModel(BaseModel):
        design_overview: Optional[str] = None
        color_scheme: ColorSchemeModel
        typography: TypographyModel
        navigation: NavigationModel
        screens: List[ScreenModel] = []
        ui_components: List[ComponentModel] = []
        responsive_design: Optional[str] = None
        accessibility: Optional[str] = None
        animations: Optional[str] = None
        icons: Optional[str] = None

    class CycleTaskModel(BaseModel):
        id: str
        title: str
        assigned_agent: Optional[str] = None
        priority: Optional[str] = None

    class CyclePlanOutputModel(BaseModel):
        plan_name: str
        tasks: List[CycleTaskModel] = []
        risks: List[str] = []

    class QualityIssueModel(BaseModel):
        severity: Optional[str] = None
        item: Optional[str] = None
        message: Optional[str] = None

    class QualityReportOutputModel(BaseModel):
        gate_decision: str
        artifact_reviewed: List[str] = []
        checklist: Optional[Dict[str, bool]] = None
        issues: List[QualityIssueModel] = []
        required_fixes: List[str] = []
        recommendations: List[str] = []

    class SupportGovernanceOutputModel(BaseModel):
        app_documentation: Optional[str] = None
        baseline_artifacts: List[str] = []
        glossary: Optional[Dict[str, str]] = None

    class PipelinePlanOutputModel(BaseModel):
        steps: List[str] = []
        reason: Optional[str] = None

else:
    DatabaseSchemaModel = None
    RequirementsOutputModel = None
    DesignOutputModel = None
    CyclePlanOutputModel = None
    QualityReportOutputModel = None
    SupportGovernanceOutputModel = None
    PipelinePlanOutputModel = None
