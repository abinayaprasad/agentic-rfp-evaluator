"""
Generate the synthetic supplier RFP proposals used by the evaluator.

Produces five PDFs in data/sample_rfps/:
    apex_systems.pdf      strong technical + compliance, highest price, 14 months
    brightpath_tech.pdf   lowest price, 7 months, weak compliance and experience
    nexaworks.pdf         balanced, best implementation plan and support model
    orbit_digital.pdf     strong references, vague integration plan, medium price
    broken_empty.pdf      valid PDF with no extractable text (error-case demo)

All companies, people, references, certificates and figures are fictional and
created for classroom use only.

Usage (from the project root):
    py -3.10 scripts/generate_rfps.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.platypus import (
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "sample_rfps"

RFP_REFERENCE = "RFP-CTMS-2026-014"
BUYER = "Helixpharm Therapeutics (fictional sponsor)"

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
_styles = getSampleStyleSheet()
STYLE_TITLE = ParagraphStyle(
    "RfpTitle", parent=_styles["Title"], fontSize=20, leading=24, spaceAfter=4,
    textColor=colors.HexColor("#1F3A5F"),
)
STYLE_SUBTITLE = ParagraphStyle(
    "RfpSubtitle", parent=_styles["Normal"], fontSize=11, leading=14,
    textColor=colors.HexColor("#555555"), alignment=TA_CENTER, spaceAfter=14,
)
STYLE_H1 = ParagraphStyle(
    "RfpH1", parent=_styles["Heading2"], fontSize=13.5, leading=17,
    spaceBefore=12, spaceAfter=6, textColor=colors.HexColor("#1F3A5F"),
)
STYLE_H2 = ParagraphStyle(
    "RfpH2", parent=_styles["Heading3"], fontSize=11, leading=14,
    spaceBefore=8, spaceAfter=3, textColor=colors.HexColor("#333333"),
)
STYLE_BODY = ParagraphStyle(
    "RfpBody", parent=_styles["Normal"], fontSize=9.6, leading=13.2, spaceAfter=6,
)
STYLE_BULLET = ParagraphStyle(
    "RfpBullet", parent=STYLE_BODY, leftIndent=12, bulletIndent=2, spaceAfter=2,
)
STYLE_CELL = ParagraphStyle("RfpCell", parent=STYLE_BODY, fontSize=8.8, leading=11.5, spaceAfter=0)
STYLE_CELL_HEAD = ParagraphStyle(
    "RfpCellHead", parent=STYLE_CELL, fontName="Helvetica-Bold", textColor=colors.white,
)
STYLE_NOTE = ParagraphStyle(
    "RfpNote", parent=STYLE_BODY, fontSize=8.5, leading=11,
    textColor=colors.HexColor("#666666"),
)


# ---------------------------------------------------------------------------
# Content model
#   Each supplier: dict with metadata and "sections": list of (heading, blocks)
#   A block is one of:
#     ("p", text)                  paragraph
#     ("h2", text)                 sub-heading
#     ("bullets", [text, ...])     bullet list
#     ("table", [header_row, row, row, ...], col_widths_mm or None)
#     ("note", text)               small grey note
# ---------------------------------------------------------------------------

APEX = {
    "file": "apex_systems.pdf",
    "supplier": "Apex Systems",
    "product": "ApexTrial CTMS",
    "tagline": "Validated clinical operations platform for global sponsors",
    "submitted": "1 October 2026",
    "contact": "Priya Raman, Director of Life Sciences Solutions (fictional)",
    "sections": [
        ("1. Executive Summary and Understanding of the Requirement", [
            ("p", f"Apex Systems is pleased to respond to {RFP_REFERENCE} issued by {BUYER}. "
                  "We understand that the sponsor operates approximately 40 active studies across 12 "
                  "countries on a legacy on-premise CTMS supplemented by spreadsheets, and requires a "
                  "validated, cloud-hosted CTMS integrated with the eTMF, EDC and SAP finance systems, "
                  "with automated investigator site payments and full migration inside 12 months."),
            ("p", "ApexTrial CTMS has been deployed at 14 sponsors and CROs since 2017. Our proposal "
                  "prioritises integration depth, inspection readiness and data integrity. We are "
                  "transparent that our delivery timeline is 14 months rather than 12; we explain in "
                  "section 3 why we consider the additional validation time a lower-risk path for a "
                  "GxP system of record."),
            ("p", "We recognise the sponsor's key risks as: loss of audit trail during migration, "
                  "payment reconciliation errors between the CTMS and SAP, and inconsistent site "
                  "activation data across countries. Each is addressed explicitly in this response."),
        ]),
        ("2. Proposed Solution and Implementation Approach", [
            ("p", "ApexTrial is a multi-tenant SaaS CTMS with a hierarchical data model of Program, "
                  "Study, Country, Site and Subject. All configuration is metadata-driven, so country-"
                  "specific regulatory milestones, visit schedules and payment rules are configured "
                  "without code changes."),
            ("h2", "Named integrations"),
            ("bullets", [
                "eTMF: bi-directional REST API (OAuth 2.0 client credentials). Site activation documents "
                "filed in the CTMS are pushed to the eTMF with TMF Reference Model metadata; eTMF "
                "completeness status is returned to the CTMS site record. Sync interval 15 minutes.",
                "EDC: inbound CDISC ODM 1.3.2 adapter consuming subject enrolment and visit completion "
                "events nightly and on demand, driving enrolment dashboards and visit-based payments.",
                "SAP finance: outbound payment requests via SAP BAPI (BAPI_ACC_DOCUMENT_POST) with vendor "
                "master lookup, and inbound payment status via IDoc. Reconciliation report lists every "
                "unmatched transaction.",
                "Identity: SAML 2.0 single sign-on with the sponsor's identity provider; SCIM provisioning.",
            ]),
            ("h2", "Reporting"),
            ("p", "Twenty-two standard reports and dashboards (enrolment vs plan, site activation cycle "
                  "time, monitoring visit compliance, payment aging) plus a governed self-service "
                  "reporting layer. All reports carry the Part 11 audit context of the underlying data."),
        ]),
        ("3. Timeline, Team Structure and Milestones", [
            ("p", "We propose a three-wave migration over 14 months. The additional two months versus "
                  "the sponsor's target are allocated to a full performance qualification of the SAP "
                  "payment interface before any live payment is released."),
            ("table", [
                ["Wave", "Scope", "Months", "Exit criteria"],
                ["0 - Foundation", "Platform provisioning, SSO, IQ, integration design, data mapping", "1-3", "IQ approved; mapping signed off"],
                ["1 - Pilot", "4 studies in 2 countries; eTMF and EDC live; payments in parallel run", "4-7", "OQ approved; pilot UAT passed"],
                ["2 - Scale", "Remaining 36 studies migrated in 3 batches; SAP payments live", "8-12", "PQ approved; legacy read-only"],
                ["3 - Close", "Hypercare, legacy decommission, validation summary report", "13-14", "VSR signed; handover complete"],
            ], [30, 70, 18, 52]),
            ("h2", "Team"),
            ("p", "Engagement director (0.5 FTE), delivery lead (1.0), solution architect (1.0), two "
                  "integration engineers (2.0), validation lead (1.0) plus two validation analysts "
                  "during OQ/PQ, data migration lead (1.0), change and training lead (0.5). All named "
                  "team members have delivered at least two prior CTMS programmes."),
            ("h2", "Risk register (extract)"),
            ("bullets", [
                "Legacy data quality below expectation - mitigated by two trial migrations before Wave 1.",
                "SAP change freeze during sponsor year-end - mitigated by scheduling BAPI go-live in month 10.",
                "Sponsor SME availability - mitigated by a named-backup roster agreed at kick-off.",
            ]),
        ]),
        ("4. Pricing and Assumptions", [
            ("p", "Pricing is in US dollars, fixed for the five-year term. Licence is per active study "
                  "per year, which scales with the sponsor's portfolio rather than headcount."),
            ("table", [
                ["Item", "Year 1", "Years 2-5 (each)", "5-year total"],
                ["Subscription licence (40 studies x $11,000)", "$440,000", "$440,000", "$2,200,000"],
                ["Implementation and migration services", "$1,250,000", "-", "$1,250,000"],
                ["Validation services (IQ/OQ/PQ, VSR)", "$480,000", "-", "$480,000"],
                ["Integration build (eTMF, EDC, SAP)", "$390,000", "-", "$390,000"],
                ["Premium support and hypercare", "$120,000", "$90,000", "$480,000"],
                ["Total", "$2,680,000", "$530,000", "$4,800,000"],
            ], [78, 30, 32, 30]),
            ("bullets", [
                "Assumes 40 active studies; additional studies at $11,000 per study per year.",
                "Assumes sponsor provides eTMF and EDC API credentials and an SAP development client by month 2.",
                "Travel billed at cost, capped at 5% of services fees.",
                "Annual licence uplift capped at 3% after year 5.",
            ]),
        ]),
        ("5. Security, Compliance and Risk Controls", [
            ("p", "ApexTrial is designed and operated as a GxP system of record. Every create, update and "
                  "delete on regulated data records who, what, when and why in a tamper-evident audit "
                  "trail that is retained for the life of the record plus 25 years."),
            ("bullets", [
                "21 CFR Part 11 and EU Annex 11: electronic signatures with meaning, re-authentication at "
                "signing, full audit trail, and a maintained requirements traceability matrix delivered "
                "with the validation package.",
                "Validation package: IQ, OQ and PQ protocols and executed reports, requirements "
                "traceability matrix, validation summary report. Sponsor retains the right to witness "
                "OQ execution.",
                "ISO 27001:2022 certified (fictional certificate AX-27001-2024-118) and SOC 2 Type II "
                "report issued annually.",
                "GDPR: EU customer data hosted in Frankfurt with a Dublin replica; US data in Virginia "
                "with an Oregon replica. Data residency is selected per study. Standard contractual "
                "clauses and a data processing agreement are provided.",
                "HIPAA-aligned safeguards for any US subject identifiers; subject data is pseudonymised "
                "at the EDC boundary.",
                "Role-based access with segregation of duties between payment requester and approver; "
                "quarterly access reviews delivered to the sponsor.",
                "Encryption in transit (TLS 1.2+) and at rest (AES-256); annual third-party penetration test "
                "with remediation evidence available under NDA.",
            ]),
        ]),
        ("6. Support Model, Relevant Experience and References", [
            ("h2", "Support"),
            ("table", [
                ["Severity", "Definition", "Response", "Restore target"],
                ["P1", "System unavailable or regulated data at risk", "15 minutes, 24x7", "4 hours"],
                ["P2", "Major function degraded, workaround exists", "1 hour, 24x7", "1 business day"],
                ["P3", "Minor defect or question", "4 business hours", "5 business days"],
            ], [20, 70, 40, 40]),
            ("p", "Hypercare of 10 weeks after each wave with a named support lead on site two days per "
                  "week. Service availability commitment of 99.9% measured monthly, with service credits."),
            ("h2", "Experience and references (fictional)"),
            ("bullets", [
                "Northwind Biologics - global CTMS replacement, 55 studies, 18 countries, 2022-2023.",
                "Cerulean Pharma - CTMS with SAP payment integration, 30 studies, 2021.",
                "Halden Therapeutics - CTMS and eTMF integration programme, 2023.",
                "Marlowe Oncology - Phase III portfolio CTMS, inspected by a regulator in 2024 with no findings.",
                "Stratus CRO - multi-sponsor CTMS, 120 studies, 2019-2020.",
                "Verdant Vaccines - rapid CTMS deployment in 9 months, 2020.",
            ]),
            ("p", "Six reference contacts can be made available on request. Three references have hosted "
                  "regulatory inspections of ApexTrial without critical or major findings."),
        ]),
    ],
}

BRIGHTPATH = {
    "file": "brightpath_tech.pdf",
    "supplier": "BrightPath Tech",
    "product": "BrightTrial",
    "tagline": "Fast, affordable trial management in the cloud",
    "submitted": "28 September 2026",
    "contact": "Daniel Okafor, Head of Sales (fictional)",
    "sections": [
        ("1. Executive Summary and Understanding of the Requirement", [
            ("p", f"BrightPath Tech thanks {BUYER} for the opportunity to respond to {RFP_REFERENCE}. "
                  "We understand you want to move off an old on-premise CTMS onto a modern cloud "
                  "platform that connects to your document, data capture and finance systems, and to do "
                  "it quickly."),
            ("p", "BrightTrial was launched in 2023 and is already used by two growing biotech companies. "
                  "Our proposition is simple: a modern user experience, the lowest total cost in this "
                  "procurement, and a go-live in seven months. We believe speed and cost should not be "
                  "traded away for features that most teams never use."),
            ("p", "Our team has strong software engineering credentials and we are keen to grow in the "
                  "pharma market. We see this engagement as a strategic partnership and have priced it "
                  "accordingly."),
        ]),
        ("2. Proposed Solution and Implementation Approach", [
            ("p", "BrightTrial is a cloud-native application built on a modern microservices stack. It "
                  "manages studies, sites, contacts, monitoring visits and enrolment tracking with "
                  "configurable dashboards. The interface is mobile friendly and requires minimal training."),
            ("h2", "Integrations"),
            ("p", "BrightTrial integrates with all major eTMF, EDC and ERP systems through our flexible "
                  "integration layer. Standard file-based and API-based options are available. Specific "
                  "connectors for the sponsor's systems will be scoped during the discovery phase once "
                  "we have access to the technical documentation."),
            ("h2", "Site payments"),
            ("p", "Visit-based payment calculations are supported. Payment files can be exported in CSV "
                  "format for upload into the sponsor's finance system. Automated posting to SAP can be "
                  "explored as a future enhancement."),
            ("h2", "Reporting"),
            ("p", "Eight standard dashboards are included. Additional reports can be built by the sponsor "
                  "using the built-in report designer."),
        ]),
        ("3. Timeline, Team Structure and Milestones", [
            ("p", "We propose a single go-live (big-bang) approach over seven months. Running one "
                  "cut-over avoids the cost and confusion of operating two systems in parallel."),
            ("table", [
                ["Phase", "Activities", "Months"],
                ["Discovery", "Requirements workshops, integration scoping", "1"],
                ["Configure", "Study templates, dashboards, user setup", "2-3"],
                ["Migrate", "Bulk load of legacy data via CSV templates", "4-5"],
                ["Test and train", "Sponsor UAT, train-the-trainer sessions", "6"],
                ["Go-live", "Cut-over weekend, two weeks hypercare", "7"],
            ], [35, 100, 25]),
            ("h2", "Team"),
            ("p", "Project manager (1.0 FTE), two configuration consultants (2.0), one integration "
                  "developer (0.5). Validation activities are the responsibility of the sponsor; we can "
                  "recommend a third-party validation consultancy if required."),
            ("h2", "Risks"),
            ("p", "The main risk is sponsor availability for workshops and UAT. We ask for a committed "
                  "sponsor project manager from day one."),
        ]),
        ("4. Pricing and Assumptions", [
            ("p", "Pricing is in US dollars. Licence is per named user per year."),
            ("table", [
                ["Item", "Year 1", "Years 2-5 (each)", "5-year total"],
                ["Subscription licence (150 users x $1,400)", "$210,000", "$210,000", "$1,050,000"],
                ["Implementation services", "$520,000", "-", "$520,000"],
                ["Integration services (discovery estimate)", "$150,000", "-", "$150,000"],
                ["Standard support", "$45,000", "$45,000", "$225,000"],
                ["Total", "$925,000", "$255,000", "$1,945,000"],
            ], [78, 30, 32, 30]),
            ("bullets", [
                "Assumes 150 named users; additional users at $1,400 per user per year.",
                "Integration services are an estimate pending discovery; final scope may change the price.",
                "Validation, if required, is excluded and would be delivered by a third party.",
                "Data migration assumes legacy data is provided in our CSV templates by the sponsor.",
            ]),
        ]),
        ("5. Security, Compliance and Risk Controls", [
            ("p", "BrightTrial is hosted on a leading public cloud provider with industry-standard "
                  "security. Data is encrypted in transit and at rest. Access is controlled through user "
                  "roles and passwords, with optional single sign-on."),
            ("p", "Changes to key records are logged with user and timestamp. The platform is designed "
                  "with regulatory requirements in mind, and we are working towards formal certification "
                  "of our information security management system, expected within the next 18 months."),
            ("p", "Backups are taken daily and retained for 90 days. Our hosting provider maintains "
                  "physical and infrastructure security certifications."),
        ]),
        ("6. Support Model, Relevant Experience and References", [
            ("h2", "Support"),
            ("p", "Support is provided by email and web portal during business hours (09:00-18:00 UK "
                  "time, Monday to Friday). Target first response within one business day. Two weeks "
                  "of hypercare follow go-live."),
            ("h2", "Experience and references (fictional)"),
            ("bullets", [
                "Lumen Biosciences - BrightTrial deployment for 6 studies, 2024.",
                "Kestrel Bio - BrightTrial deployment for 4 studies in 2 countries, 2025.",
            ]),
            ("p", "Both customers are available as references. Our engineering team has prior experience "
                  "building SaaS products in the fintech and logistics sectors."),
        ]),
    ],
}

NEXAWORKS = {
    "file": "nexaworks.pdf",
    "supplier": "NexaWorks",
    "product": "NexaCTMS",
    "tagline": "Predictable delivery for regulated clinical operations",
    "submitted": "3 October 2026",
    "contact": "Mei-Ling Chen, Engagement Partner (fictional)",
    "sections": [
        ("1. Executive Summary and Understanding of the Requirement", [
            ("p", f"NexaWorks is pleased to submit this response to {RFP_REFERENCE} from {BUYER}. "
                  "The sponsor needs to retire a legacy on-premise CTMS and spreadsheets across about "
                  "40 studies and 12 countries, replacing them with a validated cloud CTMS integrated "
                  "with the eTMF, EDC and SAP, with automated site payments, within 12 months."),
            ("p", "NexaCTMS has been live at nine sponsors and CROs since 2018. Our differentiator is "
                  "delivery discipline: a four-wave plan with published exit criteria, a dedicated "
                  "validation lead embedded from day one, and a support model with tiered SLAs that "
                  "has been inspected twice without findings."),
            ("p", "We commit to the 12-month timeline and to a fixed-price implementation. Where we "
                  "have made assumptions, they are listed in section 4."),
        ]),
        ("2. Proposed Solution and Implementation Approach", [
            ("p", "NexaCTMS provides study, country, site, subject and visit management, monitoring "
                  "visit reports, issue tracking, enrolment forecasting and investigator payments. "
                  "Configuration is template-based so new studies are set up in hours."),
            ("h2", "Integrations"),
            ("bullets", [
                "eTMF: standard REST connector. Field mapping is agreed in a two-day workshop in Wave 0. "
                "Documents created in the CTMS are filed automatically to the eTMF.",
                "EDC: standard connector that consumes enrolment and visit data via scheduled extracts "
                "(daily) using the EDC vendor's export API.",
                "SAP: payment requests are transmitted via a standard file interface agreed with the "
                "sponsor's SAP team; payment confirmations are returned and reconciled in NexaCTMS. "
                "A direct API option is available at additional cost if the sponsor prefers.",
                "Single sign-on via SAML 2.0.",
            ]),
            ("h2", "Reporting"),
            ("p", "Fifteen standard reports plus an embedded analytics module with role-based access. "
                  "Enrolment, activation and payment dashboards are refreshed hourly."),
        ]),
        ("3. Timeline, Team Structure and Milestones", [
            ("p", "We propose four waves across 12 months. Each wave has a defined exit gate that the "
                  "sponsor approves before the next wave starts. Legacy data is migrated by study "
                  "cohort with automated reconciliation reports."),
            ("table", [
                ["Wave", "Scope", "Months", "Milestone / exit gate"],
                ["0 - Mobilise", "Governance, environments, SSO, integration mapping workshops, IQ", "1-2", "M1: IQ approved, mapping signed"],
                ["1 - Pilot", "5 studies, 3 countries; eTMF and EDC connectors live; payments in shadow mode", "3-5", "M2: OQ approved, pilot UAT passed"],
                ["2 - Scale A", "18 studies migrated; SAP payments live for pilot studies", "6-8", "M3: PQ cycle 1 approved"],
                ["3 - Scale B", "Remaining 17 studies; all payments live; legacy to read-only", "9-11", "M4: PQ cycle 2 approved"],
                ["4 - Close", "Hypercare, validation summary report, legacy decommission", "12", "M5: VSR signed, handover"],
            ], [28, 70, 18, 54]),
            ("h2", "Team"),
            ("p", "Engagement partner (0.3 FTE), programme manager (1.0), solution lead (1.0), two "
                  "configuration consultants (2.0), integration engineer (1.0), dedicated validation "
                  "lead (1.0) with one analyst, data migration specialist (1.0), change management and "
                  "training lead (1.0). A weekly steering committee and a fortnightly risk review are "
                  "built into the plan."),
            ("h2", "Risk register (extract)"),
            ("bullets", [
                "Integration mapping gaps - mitigated by Wave 0 workshops with the eTMF and EDC vendors.",
                "Legacy data quality - mitigated by profiling in month 1 and two rehearsal migrations.",
                "Change fatigue across 12 countries - mitigated by country champions and localised training.",
                "Validation effort creep - mitigated by a risk-based CSV approach agreed with sponsor QA in Wave 0.",
            ]),
        ]),
        ("4. Pricing and Assumptions", [
            ("p", "Pricing is in US dollars and fixed for the term. Licence is a platform fee covering "
                  "up to 50 active studies and unlimited users."),
            ("table", [
                ["Item", "Year 1", "Years 2-5 (each)", "5-year total"],
                ["Platform subscription (up to 50 studies)", "$300,000", "$300,000", "$1,500,000"],
                ["Implementation and migration (fixed price)", "$820,000", "-", "$820,000"],
                ["Validation services (risk-based CSV)", "$260,000", "-", "$260,000"],
                ["Integration services (3 standard connectors)", "$180,000", "-", "$180,000"],
                ["Support (Gold tier) and hypercare", "$140,000", "$75,000", "$440,000"],
                ["Total", "$1,700,000", "$375,000", "$3,200,000"],
            ], [78, 30, 32, 30]),
            ("bullets", [
                "Assumes up to 50 active studies; studies 51+ at $6,000 per study per year.",
                "Assumes the sponsor's SAP team builds the receiving side of the file interface.",
                "Direct SAP API integration available for an additional $120,000 if selected.",
                "Travel included in the fixed price.",
            ]),
        ]),
        ("5. Security, Compliance and Risk Controls", [
            ("p", "NexaCTMS is operated under a quality management system aligned to GAMP 5 and has "
                  "been inspected by regulators at two customers without findings."),
            ("bullets", [
                "21 CFR Part 11 and EU Annex 11 compliant audit trail and electronic signatures on "
                "regulated records; audit trail review reports available to sponsor QA.",
                "Risk-based computer system validation led by our dedicated validation lead: IQ, OQ and "
                "PQ protocols, executed evidence and a validation summary report delivered per wave.",
                "ISO 27001:2022 certified (fictional certificate NX-ISMS-2023-041). SOC 2 Type II report "
                "available under NDA.",
                "GDPR: hosting available in the EU (Netherlands) or US (Ohio); the sponsor selects the "
                "region at contract. Data processing agreement and standard contractual clauses included.",
                "Role-based access control, segregation of duties for payments, MFA enforced for "
                "administrators, encryption in transit and at rest.",
                "Annual penetration test; summary findings shared with customers.",
            ]),
        ]),
        ("6. Support Model, Relevant Experience and References", [
            ("h2", "Support (Gold tier included)"),
            ("table", [
                ["Severity", "Definition", "Response", "Restore target"],
                ["P1", "Production down or data integrity issue", "30 minutes, 24x7", "4 hours"],
                ["P2", "Major function impaired", "2 hours, 24x7", "1 business day"],
                ["P3", "Minor issue or how-to question", "8 business hours", "5 business days"],
                ["P4", "Enhancement request", "2 business days", "Roadmap review"],
            ], [20, 70, 40, 40]),
            ("p", "Eight weeks of hypercare after each scale wave with a named customer success manager. "
                  "Availability commitment 99.8% with service credits. Quarterly service reviews."),
            ("h2", "Experience and references (fictional)"),
            ("bullets", [
                "Tidewater Pharma - CTMS replacement, 38 studies, 10 countries, delivered in 11 months, 2023.",
                "Juniper Biotherapeutics - CTMS with payment automation, 2022.",
                "Aldermoor CRO - multi-sponsor CTMS, 2021.",
                "Solvay Ridge Pharmaceuticals - CTMS and eTMF connector programme, 2024.",
                "Ironbark Health - CTMS deployment across 8 countries, 2020.",
            ]),
            ("p", "Five references available. Two have hosted regulatory inspections covering NexaCTMS "
                  "with no findings."),
        ]),
    ],
}

ORBIT = {
    "file": "orbit_digital.pdf",
    "supplier": "Orbit Digital",
    "product": "OrbitTrials CTMS",
    "tagline": "Ten years of clinical trial management expertise",
    "submitted": "3 October 2026",
    "contact": "Tomas Lindqvist, Vice President Client Services (fictional)",
    "sections": [
        ("1. Executive Summary and Understanding of the Requirement", [
            ("p", f"Orbit Digital welcomes the opportunity to respond to {RFP_REFERENCE} from {BUYER}. "
                  "We understand the sponsor seeks to modernise clinical trial management across roughly "
                  "40 studies and 12 countries, connecting the CTMS to document, data capture and finance "
                  "systems and automating investigator payments."),
            ("p", "Orbit Digital has delivered OrbitTrials CTMS to 15 pharmaceutical sponsors over ten "
                  "years. Our track record in the industry is our strongest asset: our consultants have "
                  "seen most migration pitfalls before and our client retention rate is 93%."),
            ("p", "We propose a 13-month programme at a competitive mid-range price point, backed by a "
                  "mature support organisation."),
        ]),
        ("2. Proposed Solution and Implementation Approach", [
            ("p", "OrbitTrials provides comprehensive study, site, subject, monitoring and payment "
                  "management with configurable workflows. The platform has evolved over a decade of "
                  "client feedback and covers the full clinical operations lifecycle."),
            ("h2", "Integrations"),
            ("p", "OrbitTrials uses standard connectors to integrate with common clinical and enterprise "
                  "systems. Integration with the sponsor's eTMF, EDC and finance systems will be achieved "
                  "using these standard connectors, configured during implementation. Our experience "
                  "across many client environments means most integration scenarios are familiar to us."),
            ("h2", "Site payments"),
            ("p", "Investigator payments are calculated from visit and procedure data and can be "
                  "transmitted to the finance system through the standard connector. Approval "
                  "workflows are configurable."),
            ("h2", "Reporting"),
            ("p", "Over thirty standard reports developed with clients over ten years, plus an ad-hoc "
                  "reporting tool."),
        ]),
        ("3. Timeline, Team Structure and Milestones", [
            ("p", "We propose two waves over 13 months. Wave 1 establishes the platform and migrates "
                  "half the portfolio; Wave 2 completes migration and payments."),
            ("table", [
                ["Wave", "Scope", "Months", "Key outputs"],
                ["1 - Establish", "Platform setup, configuration, integration configuration, migrate 20 studies, training", "1-7", "Platform live for 20 studies"],
                ["2 - Complete", "Migrate remaining 20 studies, payments live, hypercare, legacy decommission", "8-13", "Full portfolio live; legacy retired"],
            ], [30, 78, 18, 44]),
            ("h2", "Team"),
            ("p", "Client partner (0.3 FTE), programme manager (1.0), two senior consultants (2.0), "
                  "integration consultant (1.0), data migration consultant (1.0), training consultant "
                  "(0.5). Our consultants average nine years of CTMS experience."),
            ("h2", "Risks"),
            ("bullets", [
                "Sponsor data readiness - mitigated by our proven migration templates.",
                "Integration complexity - mitigated by our standard connector library and prior experience.",
            ]),
        ]),
        ("4. Pricing and Assumptions", [
            ("p", "Pricing is in US dollars. Licence is per active study per year."),
            ("table", [
                ["Item", "Year 1", "Years 2-5 (each)", "5-year total"],
                ["Subscription licence (40 studies x $7,500)", "$300,000", "$300,000", "$1,500,000"],
                ["Implementation and migration services", "$780,000", "-", "$780,000"],
                ["Validation support", "$150,000", "-", "$150,000"],
                ["Integration configuration (standard connectors)", "$170,000", "-", "$170,000"],
                ["Standard support", "$100,000", "$75,000", "$400,000"],
                ["Total", "$1,500,000", "$375,000", "$3,000,000"],
            ], [78, 30, 32, 30]),
            ("bullets", [
                "Assumes 40 active studies; additional studies at $7,500 per study per year.",
                "Assumes integrations are achievable with standard connectors; custom development, if "
                "needed, is quoted separately at $1,600 per day.",
                "Travel billed at cost.",
            ]),
        ]),
        ("5. Security, Compliance and Risk Controls", [
            ("p", "OrbitTrials has supported regulated clinical operations for ten years and is built "
                  "to meet applicable regulatory requirements including 21 CFR Part 11, with audit "
                  "trails and electronic signatures on key records."),
            ("bullets", [
                "Validation: we provide a validation starter pack (IQ/OQ templates) and support the "
                "sponsor's validation team; PQ is typically performed by the sponsor.",
                "ISO 27001 certified (fictional certificate OD-27001-2022-077).",
                "GDPR: we comply with applicable data protection regulations and offer a data processing "
                "agreement. Hosting is in US data centres; EU hosting can be discussed.",
                "Role-based access, encryption in transit and at rest, daily backups retained for one year.",
                "Annual third-party security assessment.",
            ]),
        ]),
        ("6. Support Model, Relevant Experience and References", [
            ("h2", "Support"),
            ("p", "24x7 support for P1 incidents with a one-hour response target; P2 and below handled "
                  "in business hours with a four-hour response target. Four weeks of hypercare after each "
                  "wave. Availability commitment 99.5%."),
            ("h2", "Experience and references (fictional)"),
            ("p", "Fifteen pharmaceutical sponsors have deployed OrbitTrials over the past ten years, "
                  "including:"),
            ("bullets", [
                "Greywell Pharma - 60 studies, 2016, still in production.",
                "Bellhaven Biologics - 25 studies, 2017.",
                "Sable Ridge Therapeutics - 45 studies, 2018.",
                "Carrow Oncology - 30 studies, 2019.",
                "Linden Vaccines - 20 studies, 2020.",
                "Whitlock Pharma - 35 studies, 2021.",
                "Faraday Biosciences - 15 studies, 2022.",
                "Oakhurst Therapeutics - 50 studies, 2023.",
                "Plus seven further sponsors available on request.",
            ]),
            ("p", "Any of these clients may be contacted as references. Several have hosted regulatory "
                  "inspections during their use of OrbitTrials."),
        ]),
    ],
}

SUPPLIERS = [APEX, BRIGHTPATH, NEXAWORKS, ORBIT]


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def _p(text: str, style: ParagraphStyle = STYLE_BODY) -> Paragraph:
    return Paragraph(escape(text), style)


def _table(rows: list[list[str]], col_widths_mm: list[float] | None) -> Table:
    data = []
    for i, row in enumerate(rows):
        style = STYLE_CELL_HEAD if i == 0 else STYLE_CELL
        data.append([Paragraph(escape(cell), style) for cell in row])
    widths = [w * mm for w in col_widths_mm] if col_widths_mm else None
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    is_total_row = rows[-1][0].strip().lower() == "total"
    ts = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3A5F")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F6FA")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BBBBBB")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if is_total_row:
        ts.append(("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#E3EAF3")))
    t.setStyle(TableStyle(ts))
    return t


def _render_block(block: tuple) -> list:
    kind = block[0]
    if kind == "p":
        return [_p(block[1])]
    if kind == "h2":
        return [_p(block[1], STYLE_H2)]
    if kind == "note":
        return [_p(block[1], STYLE_NOTE)]
    if kind == "bullets":
        return [Paragraph(escape(item), STYLE_BULLET, bulletText="•") for item in block[1]] + [Spacer(1, 4)]
    if kind == "table":
        return [_table(block[1], block[2]), Spacer(1, 6)]
    raise ValueError(f"Unknown block type: {kind}")


def _footer(supplier_name: str):
    def draw(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#777777"))
        canvas.drawString(
            doc.leftMargin, 12 * mm,
            f"{supplier_name} - Response to {RFP_REFERENCE} - Fictional proposal for classroom use only",
        )
        canvas.drawRightString(A4[0] - doc.rightMargin, 12 * mm, f"Page {doc.page}")
        canvas.restoreState()
    return draw


def build_supplier_pdf(spec: dict, output_dir: Path) -> Path:
    path = output_dir / spec["file"]
    doc = SimpleDocTemplate(
        str(path), pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm, bottomMargin=20 * mm,
        title=f"{spec['supplier']} - {RFP_REFERENCE}", author=spec["supplier"],
        subject="Synthetic RFP response (fictional)",
    )
    story: list = [
        _p(f"{spec['supplier']}", STYLE_TITLE),
        _p(f"{spec['product']} - {spec['tagline']}", STYLE_SUBTITLE),
        _table([
            ["RFP reference", "Issued by", "Submission date", "Contact"],
            [RFP_REFERENCE, BUYER, spec["submitted"], spec["contact"]],
        ], [40, 50, 30, 50]),
        Spacer(1, 8),
    ]
    for heading, blocks in spec["sections"]:
        section_flow = [_p(heading, STYLE_H1)]
        for b in blocks:
            section_flow.extend(_render_block(b))
        # keep heading with at least its first block
        story.append(KeepTogether(section_flow[:2]))
        story.extend(section_flow[2:])
    footer = _footer(spec["supplier"])
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return path


def build_broken_pdf(output_dir: Path) -> Path:
    """A structurally valid single-page PDF that contains no text at all."""
    path = output_dir / "broken_empty.pdf"
    c = rl_canvas.Canvas(str(path), pagesize=A4)
    c.setTitle("Scanned proposal (no text layer)")
    c.setFillColor(colors.HexColor("#DDDDDD"))
    c.rect(20 * mm, 20 * mm, 170 * mm, 257 * mm, stroke=0, fill=1)  # grey box, no text
    c.showPage()
    c.save()
    return path


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written = [build_supplier_pdf(s, OUTPUT_DIR) for s in SUPPLIERS]
    written.append(build_broken_pdf(OUTPUT_DIR))
    for p in written:
        print(f"wrote {p.relative_to(PROJECT_ROOT)}  ({p.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
