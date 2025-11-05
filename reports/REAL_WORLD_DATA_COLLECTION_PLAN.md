# 📋 Real-World Data Collection Plan
## Studio Revenue Simulator - Production Deployment

**Version:** 1.0  
**Date:** November 6, 2025  
**Purpose:** Collect actual studio operational data for model retraining  
**Timeline:** 3-6 months

---

## 🎯 EXECUTIVE SUMMARY

### Objective
Collect 2-3 years of historical operational data from real studios to retrain the revenue prediction model for production deployment.

### Why This Matters
Current model (R² = 0.9989) is trained on synthetic data. Real-world deployment requires:
- Training on actual studio operations
- Validation on real business patterns
- Realistic performance expectations (R² = 0.75-0.85)

### Expected Outcome
- Trained model reflecting real-world complexity
- Production-ready predictions with known accuracy
- Continuous improvement pipeline established

---

## 📊 DATA REQUIREMENTS

### 1. Minimum Data Requirements

| Category | Minimum | Recommended | Critical? |
|----------|---------|-------------|-----------|
| **Time Period** | 12 months | 24-36 months | ✅ Critical |
| **Number of Studios** | 3 studios | 8-12 studios | ✅ Critical |
| **Data Frequency** | Monthly | Monthly | ✅ Critical |
| **Data Completeness** | 80% complete | 95% complete | ⚠️ Important |
| **Data Accuracy** | Validated | Audited | ⚠️ Important |

### 2. Required Data Fields

#### A. Core Operational Metrics (CRITICAL ✅)

**Member Metrics:**
```yaml
- total_members: int
  Description: Total active members at month end
  Source: Membership management system
  Validation: Must be > 0, consistent with previous month
  
- new_members: int
  Description: New member sign-ups during month
  Source: Registration system
  Validation: Must be >= 0, <= total_members
  
- churned_members: int
  Description: Members who cancelled during month
  Source: Cancellation records
  Validation: Must be >= 0, <= previous month's total_members
  
- retention_rate: float (0-1)
  Description: % of members retained from previous month
  Formula: (total_members - new_members) / previous_total_members
  Validation: Should be between 0.60 and 0.95 typically
```

**Revenue Metrics:**
```yaml
- total_revenue: float
  Description: Total revenue for the month ($)
  Source: Accounting system / POS
  Validation: Must be > 0, reasonable vs. member count
  
- membership_revenue: float
  Description: Revenue from membership fees ($)
  Source: Billing system
  Validation: Should be 60-80% of total_revenue typically
  
- class_pack_revenue: float
  Description: Revenue from class packages ($)
  Source: POS system
  Validation: Should be 15-30% of total_revenue typically
  
- retail_revenue: float
  Description: Revenue from retail sales ($)
  Source: POS system
  Validation: Should be 5-15% of total_revenue typically
```

**Class & Attendance Metrics:**
```yaml
- total_classes_held: int
  Description: Number of classes offered during month
  Source: Scheduling system
  Validation: Must be > 0, typically 30-100 per month
  
- total_class_attendance: int
  Description: Total attendance across all classes
  Source: Check-in system / attendance tracking
  Validation: Should be reasonable vs. total_members
  
- class_attendance_rate: float (0-1)
  Description: Average attendance rate across classes
  Formula: total_class_attendance / (total_classes_held * class_capacity)
  Validation: Typically 0.40-0.80
```

**Pricing & Staff Metrics:**
```yaml
- avg_ticket_price: float
  Description: Average revenue per member ($)
  Formula: total_revenue / total_members
  Source: Calculated from revenue and member data
  Validation: Typically $150-$250 per month
  
- staff_count: int
  Description: Number of staff members
  Source: HR system / payroll
  Validation: Must be > 0, typically 5-15 for small-medium studio
  
- upsell_rate: float (0-1)
  Description: % of members purchasing beyond base membership
  Formula: members_with_additional_purchases / total_members
  Source: POS transaction analysis
  Validation: Typically 0.10-0.30
```

#### B. Studio Characteristics (IMPORTANT ⚠️)

**Studio Profile:**
```yaml
- studio_id: string
  Description: Unique identifier for studio
  Format: "STU001", "STU002", etc.
  
- studio_location: string
  Description: Urban, suburban, or rural
  Values: ["urban", "suburban", "rural"]
  
- studio_size_tier: string
  Description: Small, medium, or large
  Values: ["small", "medium", "large"]
  Criteria:
    - Small: < 200 members
    - Medium: 200-500 members
    - Large: > 500 members
  
- studio_price_tier: string
  Description: Low, medium, or high pricing
  Values: ["low", "medium", "high"]
  Criteria:
    - Low: avg_ticket_price < $150
    - Medium: $150-$200
    - High: > $200
```

**Temporal Information:**
```yaml
- month_year: date
  Description: Month and year of record
  Format: "YYYY-MM-DD" (first day of month)
  Example: "2023-01-01" for January 2023
  
- month_index: int (1-12)
  Description: Month number
  Source: Extracted from month_year
  
- year_index: int
  Description: Year number
  Source: Extracted from month_year
```

#### C. Future Targets (CRITICAL ✅)

**3-Month Forward Predictions:**
```yaml
- revenue_month_1: float
  Description: Total revenue 1 month ahead
  Note: This is what we're predicting
  
- revenue_month_2: float
  Description: Total revenue 2 months ahead
  
- revenue_month_3: float
  Description: Total revenue 3 months ahead
  
- member_count_month_3: int
  Description: Total members 3 months ahead
  
- retention_rate_month_3: float
  Description: Retention rate 3 months ahead
```

---

## 📥 DATA COLLECTION PROCESS

### Phase 1: Planning & Preparation (Week 1-2)

#### Week 1: Stakeholder Alignment
- [ ] **Present data collection plan** to leadership
- [ ] **Identify pilot studios** (3-5 studios recommended)
- [ ] **Assign data champions** at each studio
- [ ] **Schedule kickoff meetings** with studio managers

#### Week 2: Technical Setup
- [ ] **Review data sources** at each studio
  - Membership management system
  - Point-of-sale (POS) system
  - Scheduling/booking system
  - Accounting software
  
- [ ] **Assess data accessibility**
  - API access available?
  - Manual export required?
  - Data format (CSV, Excel, JSON)?
  
- [ ] **Create data templates** (see Appendix A)
- [ ] **Set up secure data storage** (cloud storage, database)

### Phase 2: Data Extraction (Week 3-6)

#### Week 3-4: Historical Data Pull
```
For each studio:
├── Month 1: Most recent complete month
├── Month 2-12: Previous 11 months
└── Month 13-36: Additional historical data (if available)
```

**Data Export Checklist (per studio):**
- [ ] Export membership data (CSV format preferred)
- [ ] Export revenue data (by category if possible)
- [ ] Export class/attendance data
- [ ] Export staff records
- [ ] Validate date ranges (no gaps)
- [ ] Check for missing months
- [ ] Document any known issues

#### Week 5-6: Data Validation
- [ ] **Completeness check:** All required fields present?
- [ ] **Range validation:** Values within expected ranges?
- [ ] **Consistency check:** Month-to-month logic holds?
- [ ] **Cross-validation:** Revenue = sum of categories?
- [ ] **Outlier detection:** Any suspicious values?

### Phase 3: Data Integration (Week 7-8)

#### Week 7: Data Cleaning
```python
# Example validation script
def validate_studio_data(df):
    checks = {
        'no_nulls_critical': df[CRITICAL_FIELDS].isnull().sum() == 0,
        'positive_values': (df['total_revenue'] > 0).all(),
        'retention_range': df['retention_rate'].between(0, 1).all(),
        'revenue_components': (
            df['membership_revenue'] + 
            df['class_pack_revenue'] + 
            df['retail_revenue']
        ).round(2) == df['total_revenue'].round(2)
    }
    return checks
```

**Cleaning Steps:**
- [ ] Handle missing values (interpolation vs. exclusion)
- [ ] Fix data type issues (strings to numbers, dates)
- [ ] Resolve inconsistencies (revenue components don't sum)
- [ ] Remove duplicate records
- [ ] Standardize studio identifiers

#### Week 8: Data Formatting
- [ ] **Combine all studios** into single dataset
- [ ] **Add studio characteristics** (location, size, price tier)
- [ ] **Create future targets** (shift revenue/members forward)
- [ ] **Generate train/test splits** (chronological)
- [ ] **Save processed data** in model-ready format

### Phase 4: Quality Assurance (Week 9-10)

#### Week 9: Data Profiling
```
Generate data quality report:
├── Completeness: % of fields populated
├── Accuracy: % passing validation rules
├── Consistency: % records logically consistent
├── Timeliness: Date range covered
└── Uniqueness: No duplicate records
```

**Quality Metrics:**
- [ ] Overall completeness ≥ 85%
- [ ] Critical field completeness = 100%
- [ ] Validation pass rate ≥ 90%
- [ ] Minimum 12 months per studio
- [ ] At least 3 studios with complete data

#### Week 10: Sign-off
- [ ] **Generate data summary report**
- [ ] **Review with data champions**
- [ ] **Get approval** from studio managers
- [ ] **Document data lineage** (sources, transformations)
- [ ] **Archive raw data** (before cleaning)

---

## 🔍 DATA QUALITY CHECKLIST

### Critical Quality Checks

```markdown
## Studio: _______________
## Data Champion: _______________
## Review Date: _______________

### Completeness
- [ ] All 12+ months present
- [ ] No gaps in monthly data
- [ ] All critical fields populated
- [ ] < 5% missing values in optional fields

### Accuracy
- [ ] Revenue figures match accounting records
- [ ] Member counts match membership system
- [ ] Attendance matches check-in logs
- [ ] Staff counts verified with HR

### Consistency
- [ ] Revenue components sum to total
- [ ] Retention rates are logical (0.60-0.95)
- [ ] New members + retained = total members
- [ ] Month-over-month changes make sense

### Validation Rules
- [ ] total_revenue > $5,000 (typical minimum)
- [ ] total_members > 50 (typical minimum)
- [ ] retention_rate between 0.60 and 0.95
- [ ] class_attendance_rate between 0.30 and 0.90
- [ ] avg_ticket_price between $100 and $300

### Edge Cases
- [ ] Outliers documented (e.g., grand opening month)
- [ ] Seasonality patterns noted (e.g., January spike)
- [ ] External events documented (e.g., COVID closures)
- [ ] Promotions/discounts noted

### Sign-off
- [ ] Data reviewed by studio manager
- [ ] Data approved by data science team
- [ ] Issues documented and resolved
- [ ] Ready for model training

Approved by: _______________ Date: _______________
```

---

## 🛠️ TOOLS & RESOURCES

### Data Collection Templates

**Template 1: Monthly Studio Data (CSV)**
```csv
studio_id,month_year,total_members,new_members,churned_members,retention_rate,avg_ticket_price,total_revenue,membership_revenue,class_pack_revenue,retail_revenue,total_classes_held,total_class_attendance,class_attendance_rate,staff_count,upsell_rate
STU001,2023-01-01,150,20,15,0.78,175.50,26325,19750,5275,1300,45,680,0.65,7,0.15
STU001,2023-02-01,155,18,13,0.82,180.00,27900,21000,5600,1300,48,720,0.68,7,0.17
```

**Template 2: Studio Characteristics (CSV)**
```csv
studio_id,studio_name,studio_location,studio_size_tier,studio_price_tier,opening_date,city,state
STU001,Studio Alpha,urban,medium,high,2020-01-15,San Francisco,CA
STU002,Studio Beta,suburban,large,medium,2019-06-01,Austin,TX
```

### Data Validation Script
```python
# save as: scripts/validate_studio_data.py

import pandas as pd
import numpy as np
from datetime import datetime

def validate_studio_data(file_path):
    """Validate studio data file"""
    df = pd.read_csv(file_path)
    
    issues = []
    
    # Check required columns
    required_cols = [
        'studio_id', 'month_year', 'total_members', 'total_revenue',
        'membership_revenue', 'class_pack_revenue', 'retail_revenue'
    ]
    
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
    
    # Check for nulls in critical fields
    for col in required_cols:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                issues.append(f"{col}: {null_count} null values")
    
    # Validate ranges
    if 'retention_rate' in df.columns:
        invalid = (~df['retention_rate'].between(0, 1)).sum()
        if invalid > 0:
            issues.append(f"retention_rate: {invalid} values outside [0,1]")
    
    # Check revenue components
    if all(col in df.columns for col in ['membership_revenue', 'class_pack_revenue', 
                                          'retail_revenue', 'total_revenue']):
        revenue_sum = (df['membership_revenue'] + 
                      df['class_pack_revenue'] + 
                      df['retail_revenue']).round(2)
        total_rev = df['total_revenue'].round(2)
        
        mismatches = (revenue_sum != total_rev).sum()
        if mismatches > 0:
            issues.append(f"Revenue components don't sum: {mismatches} rows")
    
    # Generate report
    print("\n" + "="*60)
    print(f"VALIDATION REPORT: {file_path}")
    print("="*60)
    print(f"Total Rows: {len(df)}")
    print(f"Date Range: {df['month_year'].min()} to {df['month_year'].max()}")
    print(f"Studios: {df['studio_id'].nunique()}")
    
    if issues:
        print(f"\n⚠️  {len(issues)} ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("\n✅ NO ISSUES FOUND - Data looks good!")
    
    print("="*60 + "\n")
    
    return len(issues) == 0

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        validate_studio_data(sys.argv[1])
    else:
        print("Usage: python validate_studio_data.py <data_file.csv>")
```

---

## 📅 PROJECT TIMELINE

### Gantt Chart Overview

```
Week 1-2:   Planning & Preparation      [████████░░░░░░░░░░░░░░░░]
Week 3-6:   Data Extraction             [░░░░░░░░████████████░░░░]
Week 7-8:   Data Integration            [░░░░░░░░░░░░░░░░████░░░░]
Week 9-10:  Quality Assurance           [░░░░░░░░░░░░░░░░░░░░████]
Week 11-12: Model Retraining (Next Phase)
```

### Milestones

| Week | Milestone | Deliverable | Owner |
|------|-----------|-------------|-------|
| 2 | Planning Complete | Data collection templates, studio assignments | PM |
| 6 | Data Extraction Complete | Raw data files from all studios | Data Champions |
| 8 | Data Integration Complete | Combined, cleaned dataset | Data Engineer |
| 10 | QA Complete | Validated dataset + quality report | Data Science |
| 12 | Model Retraining Complete | Production-ready model | Data Science |

---

## 👥 ROLES & RESPONSIBILITIES

### Project Team

**Project Manager**
- Coordinate overall data collection effort
- Track progress against timeline
- Resolve blockers and escalations
- Communicate status to stakeholders

**Data Engineer**
- Design data collection templates
- Build data validation scripts
- Integrate data from multiple studios
- Ensure data quality and consistency

**Data Scientist**
- Define data requirements
- Validate data suitability for modeling
- Retrain model on real data
- Evaluate performance metrics

**Studio Data Champions** (1 per studio)
- Extract data from studio systems
- Validate accuracy with studio manager
- Communicate any data issues
- Provide context for unusual patterns

**IT Support**
- Assist with system access
- Troubleshoot data export issues
- Set up secure data transfer
- Ensure compliance with data policies

---

## 🔒 DATA SECURITY & PRIVACY

### Security Measures

**Data Transmission:**
- [ ] Use secure file transfer (SFTP, encrypted email)
- [ ] Password-protect all data files
- [ ] Use VPN for remote access
- [ ] Delete data from insecure locations after transfer

**Data Storage:**
- [ ] Store in secure cloud storage (AWS S3, Azure, etc.)
- [ ] Implement access controls (need-to-know basis)
- [ ] Enable encryption at rest
- [ ] Regular backups with version control

**Data Privacy:**
- [ ] Anonymize studio identifiers if needed
- [ ] Remove any PII (member names, emails, etc.)
- [ ] Comply with data retention policies
- [ ] Document data usage and retention

### Compliance Checklist
- [ ] GDPR compliance (if applicable)
- [ ] CCPA compliance (if applicable)
- [ ] Internal data governance policies
- [ ] Data processing agreement signed
- [ ] Privacy impact assessment completed

---

## 🚨 RISK MANAGEMENT

### Potential Risks & Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **Incomplete data** | High | Medium | Start with 5+ studios, need only 3 complete |
| **Data quality issues** | High | Medium | Robust validation, work with studios to fix |
| **System access delays** | Medium | High | Start access requests early, escalate if needed |
| **Studio resistance** | Medium | Low | Clear communication of benefits, executive support |
| **Timeline delays** | Low | Medium | Build in 2-week buffer, phase releases |
| **Data security incident** | High | Low | Follow security protocols, train team |

---

## 📊 SUCCESS METRICS

### Data Collection Success Criteria

| Metric | Target | Stretch Goal |
|--------|--------|--------------|
| **Number of Studios** | 3 | 8-12 |
| **Time Period** | 12 months | 24-36 months |
| **Data Completeness** | 85% | 95% |
| **Data Accuracy** | 90% validation pass | 98% validation pass |
| **Timeline** | 10 weeks | 8 weeks |

### Model Performance Success Criteria (Post-Retraining)

| Metric | Minimum Acceptable | Target | Current (Synthetic) |
|--------|-------------------|--------|---------------------|
| **R² Score** | 0.70 | 0.75-0.85 | 0.9989 |
| **MAPE** | < 20% | < 15% | 2.0% |
| **RMSE** | < $5,000 | < $3,000 | $499 |

---

## 📋 APPENDIX

### Appendix A: Data Dictionary

**Complete field definitions, data types, and validation rules**  
See: `data_dictionary.xlsx`

### Appendix B: Sample Data Files

**Example CSV files with sample data**  
See: `data/samples/`

### Appendix C: Validation Scripts

**Python scripts for data validation**  
See: `scripts/validate_studio_data.py`

### Appendix D: FAQ

**Q: What if a studio doesn't have all the data fields?**  
A: Focus on critical fields (✅). Optional fields can be estimated or left null if < 10% missing.

**Q: How do we handle studios that opened recently (< 12 months)?**  
A: Include all available data. Studios with 6+ months can still be useful.

**Q: What if there are gaps in the data (missing months)?**  
A: Document gaps. Can interpolate for 1-2 missing months, but more requires studio follow-up.

**Q: How often should we update the data?**  
A: Monthly updates recommended after initial collection. Quarterly model retraining.

**Q: What about COVID-19 impact on data?**  
A: Flag months with closures/restrictions. Can exclude from training or add indicator variable.

---

## 📞 CONTACTS & SUPPORT

**Project Lead:** [Name], [Email]  
**Data Engineering:** [Name], [Email]  
**Data Science:** [Name], [Email]  
**IT Support:** [Email/Ticket System]

**Documentation:** `docs/data_collection/`  
**Issue Tracking:** [Jira/Asana/GitHub Issues]  
**Team Chat:** [Slack/Teams Channel]

---

**Document Version:** 1.0  
**Last Updated:** November 6, 2025  
**Next Review:** Start of data collection (Week 1)

---

*This plan is a living document. Update as needed based on learnings during data collection.*

