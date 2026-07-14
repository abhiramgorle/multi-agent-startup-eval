"""
Synthetic ground-truth dataset for evaluating multi-agent startup evaluation performance.

Each case has an expected score range and recommendation based on archetypes that
any reasonable investment evaluator should consistently categorize correctly.

Tier mapping:
  Tier 1 — Strong Invest  (8.0 – 10.0)
  Tier 2 — Invest         (6.5 –  7.9)
  Tier 3 — Conditional Pass (5.0 – 6.4)
  Tier 4 — Pass           (3.5 –  4.9)
  Tier 5 — Strong Pass    (1.0 –  3.4)
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class GroundTruth:
    id: str
    name: str
    description: str
    expected_score_min: float
    expected_score_max: float
    expected_recommendation: str   # matches SynthesisJudge._score_to_recommendation
    expected_tier: int             # 1-5
    domain_tags: tuple[str, ...]
    rationale: str                 # why this score bracket is expected


# ── Tier 1: Strong Invest (8.0 – 10.0) ──────────────────────────────────────

_T1_DEVOPS_AI = GroundTruth(
    id="GT-T1-001",
    name="DevOps AI Copilot",
    description=(
        "An AI-native DevOps platform that monitors cloud infrastructure in real time, "
        "auto-diagnoses incidents using LLM reasoning over logs/metrics, and proposes "
        "one-click remediations. Targets mid-market SaaS companies spending $500K+/yr "
        "on cloud ops. Priced at $2,000/mo per cluster with usage-based upsell. "
        "Founding team: ex-AWS principal engineer + ex-PagerDuty PM. $1.2M ARR in 8 months "
        "with 94% gross retention. Integration with Datadog, Grafana, and GitHub Actions."
    ),
    expected_score_min=7.8,
    expected_score_max=10.0,
    expected_recommendation="Strong Invest",
    expected_tier=1,
    domain_tags=("SaaS", "DevOps", "AI", "B2B"),
    rationale=(
        "Clear large TAM (cloud ops market ~$50B), proven revenue traction, strong technical "
        "team pedigree, defensible workflow integrations, healthy gross retention."
    ),
)

_T1_HEALTH_RECORDS = GroundTruth(
    id="GT-T1-002",
    name="AI Clinical Documentation Assistant",
    description=(
        "Reduces physician documentation burden by 70% via ambient AI that listens to "
        "patient-doctor conversations and auto-generates structured EHR notes (SOAP format). "
        "HIPAA-compliant, SOC2 Type II certified. Integrates with Epic, Cerner, and Athena. "
        "Charging $500/physician/month. 3,200 physicians across 18 health systems. "
        "NPS of 72. Ex-Stanford Medicine + ex-Google Health founding team. "
        "FDA 510(k) clearance obtained. Saving physicians 2 hours/day on average."
    ),
    expected_score_min=7.8,
    expected_score_max=10.0,
    expected_recommendation="Strong Invest",
    expected_tier=1,
    domain_tags=("HealthTech", "AI", "B2B", "SaaS"),
    rationale=(
        "Massive documented pain point, strong traction with 3,200 paying physicians, "
        "regulatory moat (510k), EHR integrations as defensive moat, high NPS."
    ),
)

_T1_SMB_FINANCE = GroundTruth(
    id="GT-T1-003",
    name="CFO-in-a-Box for SMBs",
    description=(
        "Automated financial operations platform for SMBs ($1M-$10M revenue) that combines "
        "bookkeeping, cash flow forecasting, accounts payable automation, and tax preparation "
        "into a single $299/month subscription. Replaces a part-time CFO costing $5,000+/month. "
        "Integrates with QuickBooks, Xero, Stripe, and Square. 4,800 paying customers, "
        "$17M ARR, 118% net revenue retention. Founded by ex-Intuit and ex-Stripe engineers. "
        "Payback period under 3 months."
    ),
    expected_score_min=7.8,
    expected_score_max=10.0,
    expected_recommendation="Strong Invest",
    expected_tier=1,
    domain_tags=("FinTech", "SMB", "SaaS", "B2B"),
    rationale=(
        "30M SMB TAM, clear 10x ROI vs alternatives, strong traction, 118% NRR showing "
        "expansion revenue, short payback, defensible accounting integrations."
    ),
)

# ── Tier 2: Invest (6.5 – 7.9) ──────────────────────────────────────────────

_T2_MARKETPLACE = GroundTruth(
    id="GT-T2-001",
    name="Freelance Robotics Engineer Marketplace",
    description=(
        "Two-sided marketplace matching specialized robotics engineers (ROS, SLAM, computer vision) "
        "with manufacturing SMBs needing automation consulting. Takes 18% commission on $150-300/hr "
        "engagements. 340 vetted engineers, 180 client companies. GMV of $2.1M in year 1. "
        "Chicken-and-egg dynamic largely solved in the US Southeast manufacturing corridor. "
        "No dominant incumbent — AngelList Talent and Upwork do not vet for robotics depth. "
        "Founder has 12 years in industrial automation at Fanuc."
    ),
    expected_score_min=6.2,
    expected_score_max=8.2,
    expected_recommendation="Invest",
    expected_tier=2,
    domain_tags=("Marketplace", "Robotics", "B2B"),
    rationale=(
        "Niche but growing market, liquidity problem largely solved, credible founder, "
        "defensible vetting. Concern: geographic concentration and supply constraints."
    ),
)

_T2_EDTECH = GroundTruth(
    id="GT-T2-002",
    name="AI Coding Tutor for Bootcamp Students",
    description=(
        "Personalized AI tutoring platform for coding bootcamp students that adapts exercises "
        "in real time based on error patterns, offers Socratic debugging (never gives answers, "
        "only hints), and predicts dropout risk 2 weeks in advance. B2B model: sold to bootcamps "
        "at $50/student/month. 22 bootcamp partners, 8,400 active students. Retention improvement "
        "of 31% for partner bootcamps. Founded by two ex-Coursera ML engineers."
    ),
    expected_score_min=6.2,
    expected_score_max=8.2,
    expected_recommendation="Invest",
    expected_tier=2,
    domain_tags=("EdTech", "AI", "B2B", "SaaS"),
    rationale=(
        "Proven retention impact, B2B distribution avoids consumer CAC issues, solid traction. "
        "Risk: bootcamp industry consolidation and potential direct competition from Duolingo/Coursera."
    ),
)

_T2_SUPPLY_CHAIN = GroundTruth(
    id="GT-T2-003",
    name="Supply Chain Visibility for D2C Brands",
    description=(
        "Real-time supply chain visibility and exception management tool for direct-to-consumer "
        "brands ($5M-$50M revenue). Aggregates data from 200+ freight carriers, 3PLs, and "
        "factory portals. Alerts on delays, suggests rerouting options, and forecasts stockouts "
        "4 weeks ahead using ML on historical + live data. $1,500/month SaaS. 290 customers, "
        "$5.2M ARR. CAC of $3,200, LTV of $28,000. Team of 18 including 6 ML engineers."
    ),
    expected_score_min=6.2,
    expected_score_max=8.2,
    expected_recommendation="Invest",
    expected_tier=2,
    domain_tags=("Supply Chain", "ML", "D2C", "B2B"),
    rationale=(
        "Strong unit economics (LTV/CAC ~8.75x), real pain proven by COVID disruptions, "
        "data network effect. Risk: Flexport, project44 as well-funded incumbents."
    ),
)

# ── Tier 3: Conditional Pass (5.0 – 6.4) ────────────────────────────────────

_T3_SOCIAL = GroundTruth(
    id="GT-T3-001",
    name="Anonymous Professional Feedback Network",
    description=(
        "Social platform where professionals give and receive anonymous, structured 360-degree "
        "feedback outside of their company. Freemium model: free for 3 feedback requests/month, "
        "$12/month for unlimited. 85,000 registered users, 12,000 paying ($144K MRR). "
        "Retention drops sharply after month 3 (only 34% month-4 retention). "
        "Concerns about anonymity being gamed for workplace harassment. "
        "Founded by first-time founders, ex-LinkedIn employees."
    ),
    expected_score_min=4.5,
    expected_score_max=6.5,
    expected_recommendation="Conditional Pass",
    expected_tier=3,
    domain_tags=("SaaS", "B2C", "Social", "HR"),
    rationale=(
        "Decent traction but severe retention cliff, first-time founders, brand/legal risk "
        "from anonymity misuse, unclear defensibility vs LinkedIn adding similar features."
    ),
)

_T3_CLIMATE = GroundTruth(
    id="GT-T3-002",
    name="Carbon Credit Marketplace for SMBs",
    description=(
        "Platform allowing SMBs to purchase verified carbon offsets to reach net-zero claims, "
        "with automated carbon footprint calculation from utility bills and supply chain data. "
        "Takes 8% transaction fee. $340K in transactions in 6 months, 180 SMB customers. "
        "Regulatory risk: SEC climate disclosure rules under litigation. Market dependent on "
        "corporate ESG mandates which may weaken. Founder has 5 years in carbon markets at "
        "South Pole. Competing with Patch, Pachama, and Gold Standard."
    ),
    expected_score_min=4.5,
    expected_score_max=6.5,
    expected_recommendation="Conditional Pass",
    expected_tier=3,
    domain_tags=("ClimaTech", "Marketplace", "ESG"),
    rationale=(
        "Early traction, experienced founder, but market is regulatory-dependent, "
        "crowded with better-funded players, and ESG sentiment is politically volatile."
    ),
)

_T3_CONSUMER_APP = GroundTruth(
    id="GT-T3-003",
    name="Habit Stacking Mobile App",
    description=(
        "Consumer mobile app that bundles habit tracking, micro-journaling, and peer accountability "
        "groups. Gamified with streaks and social leaderboards. Freemium: $6.99/month premium. "
        "480,000 downloads, 2.1% paid conversion (10,080 paying), $70K MRR. "
        "D30 retention at 18% (industry average: 15-25%). High viral coefficient (K=1.3) "
        "but low LTV ($84 average lifetime) vs rising UA costs ($4.20 CPI on Meta). "
        "No unique technical moat. Competing with Habitica, Streaks, and Fabulous."
    ),
    expected_score_min=4.5,
    expected_score_max=6.5,
    expected_recommendation="Conditional Pass",
    expected_tier=3,
    domain_tags=("Consumer", "Mobile", "B2C", "Wellness"),
    rationale=(
        "Average retention, thin margins, no moat, competitive consumer market. "
        "Positive: solid viral loop and decent conversion rate. Marginal unit economics."
    ),
)

# ── Tier 4: Pass (3.5 – 4.9) ────────────────────────────────────────────────

_T4_CRYPTO = GroundTruth(
    id="GT-T4-001",
    name="NFT-Gated Luxury Concierge Service",
    description=(
        "Luxury travel and lifestyle concierge service where access is gated by ownership "
        "of an NFT from a proprietary collection (floor price: $200). Members pay $0 "
        "subscription but NFT holders receive 'exclusive' access to travel bookings, "
        "restaurant reservations, and event tickets. Revenue from referral commissions "
        "(3-5%) from travel and dining partners. 2,100 NFTs minted, 340 active users. "
        "NFT floor price dropped 82% since launch. No recurring revenue. "
        "Founders have no hospitality or Web3 track record."
    ),
    expected_score_min=2.5,
    expected_score_max=4.9,
    expected_recommendation="Pass",
    expected_tier=4,
    domain_tags=("Web3", "NFT", "Consumer", "Luxury"),
    rationale=(
        "NFT-gated access model with collapsing floor price destroys community lock-in, "
        "commission revenue insufficient to sustain operations, tiny active user base, "
        "inexperienced team, no defensibility."
    ),
)

_T4_GENERIC_AI = GroundTruth(
    id="GT-T4-002",
    name="AI-Powered General Business Consultant Chatbot",
    description=(
        "A chatbot that answers general business questions using GPT-4 API with a thin "
        "prompt engineering layer. Marketed as 'your AI business advisor.' Charges $29/month. "
        "750 paying customers. No proprietary data, no vertical specialization, no unique "
        "integrations. Founders claim the moat is 'prompt engineering expertise.' "
        "No team with relevant domain expertise. "
        "Competing directly with ChatGPT Plus, Claude.ai, Perplexity, and dozens of similar wrappers. "
        "Churn rate: 38% monthly."
    ),
    expected_score_min=2.5,
    expected_score_max=4.9,
    expected_recommendation="Pass",
    expected_tier=4,
    domain_tags=("AI", "SaaS", "B2B", "Chatbot"),
    rationale=(
        "Classic thin wrapper with no moat, 38% monthly churn is catastrophic, "
        "directly competing against OpenAI/Anthropic who control the underlying model, "
        "zero proprietary advantage."
    ),
)

_T4_LOCAL_SERVICE = GroundTruth(
    id="GT-T4-003",
    name="On-Demand Dog Walking Marketplace (Suburban Only)",
    description=(
        "Uber-for-dog-walking targeting a single suburban US city (population 180,000). "
        "Commission model at 20%. 85 registered walkers, 310 pet owners, $8,200 monthly GMV. "
        "High geographic concentration, no plan for expansion. "
        "Competing with Rover (160M users, $1.35B raised) and Wag (went public via SPAC, "
        "now struggling). Founders plan to 'own this city first' before expanding, "
        "but unit economics require 50+ cities to be sustainable. No tech differentiation."
    ),
    expected_score_min=2.5,
    expected_score_max=4.9,
    expected_recommendation="Pass",
    expected_tier=4,
    domain_tags=("Marketplace", "Consumer", "Local Services"),
    rationale=(
        "Hyper-local with dominant well-funded incumbents, no tech moat, unit economics "
        "require unproven multi-city expansion, tiny GMV relative to operational complexity."
    ),
)

# ── Tier 5: Strong Pass (1.0 – 3.4) ─────────────────────────────────────────

_T5_PERPETUAL_MOTION = GroundTruth(
    id="GT-T5-001",
    name="Quantum Energy Harvesting Device",
    description=(
        "Hardware device that claims to harvest 'ambient quantum energy' from the environment "
        "to generate electricity without any external power source. Claims 300% energy efficiency "
        "(more output than input). Seeking $5M Series A to manufacture and distribute. "
        "No peer-reviewed research, no prototype demonstrated to third parties. "
        "Founder has a background in 'alternative physics' with no engineering credentials. "
        "Patent applications filed but no approvals. Pre-revenue."
    ),
    expected_score_min=1.0,
    expected_score_max=3.0,
    expected_recommendation="Strong Pass",
    expected_tier=5,
    domain_tags=("Hardware", "CleanTech", "Science"),
    rationale=(
        "Violates laws of thermodynamics (perpetual motion machine). No credible prototype, "
        "no scientific validation, fraudulent energy efficiency claims. Not investable."
    ),
)

_T5_NEGATIVE_TAM = GroundTruth(
    id="GT-T5-002",
    name="Physical Fax Machine Rental for Gen Z",
    description=(
        "Subscription service renting retro fax machines to Gen Z consumers as a 'nostalgic "
        "analog communication experience.' $49/month rental including paper and toner. "
        "Targeting 18-26 year olds who 'want to disconnect from digital.' "
        "12 paying customers after 8 months of operation. Marketing on TikTok. "
        "Fax machine supply from liquidated office equipment. "
        "No B2B strategy. Founder believes this is 'the vinyl records of communication.'"
    ),
    expected_score_min=1.0,
    expected_score_max=3.0,
    expected_recommendation="Strong Pass",
    expected_tier=5,
    domain_tags=("Hardware", "Consumer", "Rental"),
    rationale=(
        "Negligible TAM, 12 customers after 8 months proves no product-market fit, "
        "nostalgia thesis unproven at this price point, zero defensibility, "
        "operational complexity for near-zero revenue."
    ),
)

_T5_LEGAL_ISSUE = GroundTruth(
    id="GT-T5-003",
    name="Scraped Social Media Data Reseller",
    description=(
        "Platform that scrapes public social media profiles (LinkedIn, Instagram, Twitter/X) "
        "without authorization to build enriched contact databases sold to sales teams. "
        "Charging $299/month for 10,000 contacts/month. 420 paying customers, $125K MRR. "
        "Received cease-and-desist letters from LinkedIn and Meta. Founders claim "
        "'public data is fair use' and plan to fight in court. "
        "GDPR and CCPA compliance not addressed. Prior similar companies (hiQ Labs) "
        "faced decade-long litigation."
    ),
    expected_score_min=1.0,
    expected_score_max=3.0,
    expected_recommendation="Strong Pass",
    expected_tier=5,
    domain_tags=("Data", "SaaS", "B2B", "Legal Risk"),
    rationale=(
        "Active legal exposure from platform ToS violations and GDPR/CCPA non-compliance, "
        "C&D letters already received, business model may be enjoined at any time, "
        "existential legal risk regardless of revenue traction."
    ),
)


# ── Full dataset ─────────────────────────────────────────────────────────────

GROUND_TRUTH_DATASET: list[GroundTruth] = [
    _T1_DEVOPS_AI,
    _T1_HEALTH_RECORDS,
    _T1_SMB_FINANCE,
    _T2_MARKETPLACE,
    _T2_EDTECH,
    _T2_SUPPLY_CHAIN,
    _T3_SOCIAL,
    _T3_CLIMATE,
    _T3_CONSUMER_APP,
    _T4_CRYPTO,
    _T4_GENERIC_AI,
    _T4_LOCAL_SERVICE,
    _T5_PERPETUAL_MOTION,
    _T5_NEGATIVE_TAM,
    _T5_LEGAL_ISSUE,
]
