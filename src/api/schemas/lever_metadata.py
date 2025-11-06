"""
Centralized Lever Metadata Registry

This module provides a single source of truth for all business lever definitions,
including constraints, descriptions, priorities, and implementation guidance.
"""

from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field


class FeasibilityThresholds(BaseModel):
    """Difficulty thresholds for implementing lever changes"""
    easy: float = Field(..., description="Changes below this threshold are easy to implement")
    moderate: float = Field(..., description="Changes below this threshold are moderately difficult")
    hard: float = Field(..., description="Changes below this threshold are hard to implement")
    very_hard: float = Field(..., description="Changes above 'hard' threshold are very difficult")


class ActionTemplate(BaseModel):
    """Implementation guidance for lever adjustments"""
    description: str = Field(..., description="General action description")
    timeline_weeks: int = Field(..., description="Typical implementation timeline in weeks")
    department: str = Field(..., description="Department or role responsible")
    resources: Optional[str] = Field(None, description="Resources needed for implementation")


class LeverConstraints(BaseModel):
    """Min/max constraints for a lever"""
    min: float = Field(..., description="Minimum allowed value")
    max: float = Field(..., description="Maximum allowed value")


class LeverMetadata(BaseModel):
    """Complete metadata for a business lever"""
    name: str = Field(..., description="Lever identifier (snake_case)")
    display_name: str = Field(..., description="Human-readable name")
    data_type: Literal["float", "integer"] = Field(..., description="Data type")
    constraints: LeverConstraints = Field(..., description="Min/max value constraints")
    description: str = Field(..., description="Detailed description of the lever")
    unit: str = Field(..., description="Unit of measurement (percentage, dollars, count)")
    default_value: Optional[float] = Field(None, description="Suggested default value")
    priority: int = Field(..., ge=1, le=8, description="Priority ranking (1=highest, 8=lowest)")
    feasibility_thresholds: FeasibilityThresholds = Field(..., description="Difficulty thresholds for changes")
    action_template: ActionTemplate = Field(..., description="Implementation guidance")


# Centralized registry of all business levers
LEVER_REGISTRY: Dict[str, LeverMetadata] = {
    "retention_rate": LeverMetadata(
        name="retention_rate",
        display_name="Member Retention Rate",
        data_type="float",
        constraints=LeverConstraints(min=0.5, max=1.0),
        description="Percentage of members who remain active from one month to the next",
        unit="percentage",
        default_value=0.75,
        priority=1,
        feasibility_thresholds=FeasibilityThresholds(
            easy=0.02,
            moderate=0.05,
            hard=0.10,
            very_hard=0.10
        ),
        action_template=ActionTemplate(
            description="Improve member retention through enhanced engagement programs and personalized services",
            timeline_weeks=8,
            department="Member Success",
            resources="Engagement software, customer success team"
        )
    ),
    "avg_ticket_price": LeverMetadata(
        name="avg_ticket_price",
        display_name="Average Ticket Price",
        data_type="float",
        constraints=LeverConstraints(min=50.0, max=500.0),
        description="Average monthly membership or package price per customer",
        unit="dollars",
        default_value=150.0,
        priority=2,
        feasibility_thresholds=FeasibilityThresholds(
            easy=5.0,
            moderate=20.0,
            hard=50.0,
            very_hard=50.0
        ),
        action_template=ActionTemplate(
            description="Implement strategic pricing increase with value-added services",
            timeline_weeks=4,
            department="Pricing & Revenue",
            resources="Market analysis, pricing strategy consultant"
        )
    ),
    "class_attendance_rate": LeverMetadata(
        name="class_attendance_rate",
        display_name="Class Attendance Rate",
        data_type="float",
        constraints=LeverConstraints(min=0.4, max=1.0),
        description="Percentage of registered members who attend classes regularly",
        unit="percentage",
        default_value=0.70,
        priority=5,
        feasibility_thresholds=FeasibilityThresholds(
            easy=0.03,
            moderate=0.07,
            hard=0.15,
            very_hard=0.15
        ),
        action_template=ActionTemplate(
            description="Improve class attendance through better scheduling and member engagement",
            timeline_weeks=6,
            department="Operations",
            resources="Scheduling software, communication tools"
        )
    ),
    "new_members": LeverMetadata(
        name="new_members",
        display_name="New Members per Month",
        data_type="integer",
        constraints=LeverConstraints(min=0, max=100),
        description="Number of new members acquired per month",
        unit="count",
        default_value=25,
        priority=3,
        feasibility_thresholds=FeasibilityThresholds(
            easy=5,
            moderate=15,
            hard=30,
            very_hard=30
        ),
        action_template=ActionTemplate(
            description="Increase new member acquisition through marketing and referral programs",
            timeline_weeks=12,
            department="Marketing & Sales",
            resources="Marketing budget, sales team training"
        )
    ),
    "staff_utilization_rate": LeverMetadata(
        name="staff_utilization_rate",
        display_name="Staff Utilization Rate",
        data_type="float",
        constraints=LeverConstraints(min=0.6, max=1.0),
        description="Percentage of staff time spent on revenue-generating activities",
        unit="percentage",
        default_value=0.85,
        priority=6,
        feasibility_thresholds=FeasibilityThresholds(
            easy=0.05,
            moderate=0.10,
            hard=0.20,
            very_hard=0.20
        ),
        action_template=ActionTemplate(
            description="Optimize staff scheduling and reduce non-productive time",
            timeline_weeks=4,
            department="Operations",
            resources="Scheduling optimization software"
        )
    ),
    "upsell_rate": LeverMetadata(
        name="upsell_rate",
        display_name="Upsell Rate",
        data_type="float",
        constraints=LeverConstraints(min=0.0, max=0.5),
        description="Percentage of members who purchase additional services or upgrades",
        unit="percentage",
        default_value=0.25,
        priority=4,
        feasibility_thresholds=FeasibilityThresholds(
            easy=0.03,
            moderate=0.08,
            hard=0.15,
            very_hard=0.15
        ),
        action_template=ActionTemplate(
            description="Train staff on upselling techniques and create attractive package bundles",
            timeline_weeks=6,
            department="Sales & Training",
            resources="Sales training program, package design"
        )
    ),
    "total_classes_held": LeverMetadata(
        name="total_classes_held",
        display_name="Total Classes per Month",
        data_type="integer",
        constraints=LeverConstraints(min=50, max=500),
        description="Total number of classes offered per month",
        unit="count",
        default_value=120,
        priority=7,
        feasibility_thresholds=FeasibilityThresholds(
            easy=10,
            moderate=30,
            hard=60,
            very_hard=60
        ),
        action_template=ActionTemplate(
            description="Adjust class schedule to optimize capacity and demand",
            timeline_weeks=2,
            department="Operations",
            resources="Scheduling software, instructor availability"
        )
    ),
    "total_members": LeverMetadata(
        name="total_members",
        display_name="Total Member Count",
        data_type="integer",
        constraints=LeverConstraints(min=50, max=1000),
        description="Current total number of active members",
        unit="count",
        default_value=200,
        priority=8,
        feasibility_thresholds=FeasibilityThresholds(
            easy=10,
            moderate=30,
            hard=60,
            very_hard=60
        ),
        action_template=ActionTemplate(
            description="Grow member base through combined acquisition and retention strategies",
            timeline_weeks=16,
            department="Marketing & Member Success",
            resources="Marketing budget, retention programs"
        )
    )
}


def get_all_levers(include_details: bool = False) -> List[Dict]:
    """
    Get all levers with their metadata

    Args:
        include_details: If True, include feasibility thresholds and action templates

    Returns:
        List of lever metadata dictionaries
    """
    levers = []

    for lever_name, metadata in LEVER_REGISTRY.items():
        lever_dict = {
            "name": metadata.name,
            "display_name": metadata.display_name,
            "data_type": metadata.data_type,
            "constraints": metadata.constraints.dict(),
            "description": metadata.description,
            "unit": metadata.unit,
            "default_value": metadata.default_value,
            "priority": metadata.priority
        }

        if include_details:
            lever_dict["feasibility_thresholds"] = metadata.feasibility_thresholds.dict()
            lever_dict["action_template"] = metadata.action_template.dict()

        levers.append(lever_dict)

    # Sort by priority
    levers.sort(key=lambda x: x["priority"])

    return levers


def get_lever_by_name(lever_name: str) -> Optional[LeverMetadata]:
    """
    Get metadata for a specific lever

    Args:
        lever_name: Name of the lever

    Returns:
        LeverMetadata or None if not found
    """
    return LEVER_REGISTRY.get(lever_name)


def get_lever_constraints() -> Dict[str, tuple]:
    """
    Get lever constraints in (min, max) tuple format

    Used for backward compatibility with existing services

    Returns:
        Dictionary mapping lever names to (min, max) tuples
    """
    return {
        name: (metadata.constraints.min, metadata.constraints.max)
        for name, metadata in LEVER_REGISTRY.items()
    }
