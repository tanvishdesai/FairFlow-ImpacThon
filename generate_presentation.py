"""
FairFlow Hackathon Presentation Generator
Generates HTML slides and converts them to PDF
"""

import os
from pathlib import Path

# Slide content for the presentation
SLIDES = [
    # Slide 1: Problem Statement
    {
        "title": "PROBLEM STATEMENT",
        "content": """
        <div class="form-grid">
            <div class="form-row">
                <span class="label">Track Number:</span>
                <span class="value">2</span>
            </div>
            <div class="form-row">
                <span class="label">Track Title:</span>
                <span class="value">Safe, Trusted & Responsible Technology</span>
            </div>
            <div class="form-row">
                <span class="label">Video Link:</span>
                <span class="value">(Optional)</span>
            </div>
            <div class="form-row">
                <span class="label">Team ID:</span>
                <span class="value">__________</span>
            </div>
            <div class="form-row">
                <span class="label">Team Name:</span>
                <span class="value">__________</span>
            </div>
            <div class="form-row">
                <span class="label">Institute Name:</span>
                <span class="value">__________</span>
            </div>
        </div>
        <div class="project-title">
            <h2>FairFlow: The RL-Driven Adaptive Bias Firewall</h2>
            <p class="tagline">An Enterprise AI Governance Platform for Real-Time Fairness Compliance</p>
        </div>
        """,
        "slide_num": 1
    },
    
    # Slide 2: Team Details
    {
        "title": "TEAM DETAILS",
        "content": """
        <div class="team-grid">
            <div class="team-member">
                <div class="avatar">👤</div>
                <h3>Member 1 Name</h3>
                <p class="role">Team Leader</p>
                <p class="details">Enrollment Number</p>
                <p class="details">Department, Institute Name, KSV</p>
            </div>
            <div class="team-member">
                <div class="avatar">👤</div>
                <h3>Member 2 Name</h3>
                <p class="role">Co-Team Leader</p>
                <p class="details">Enrollment Number</p>
                <p class="details">Department, Institute Name, KSV</p>
            </div>
            <div class="team-member">
                <div class="avatar">👤</div>
                <h3>Member 3 Name</h3>
                <p class="role">Team Member</p>
                <p class="details">Enrollment Number</p>
                <p class="details">Department, Institute Name, KSV</p>
            </div>
            <div class="team-member">
                <div class="avatar">👤</div>
                <h3>Member 4 Name</h3>
                <p class="role">Team Member</p>
                <p class="details">Enrollment Number</p>
                <p class="details">Department, Institute Name, KSV</p>
            </div>
            <div class="team-member">
                <div class="avatar">🎓</div>
                <h3>Guide Name</h3>
                <p class="role">Guide</p>
                <p class="details">Department, Institute Name, KSV</p>
            </div>
        </div>
        """,
        "slide_num": 2
    },
    
    # Slide 3: Idea Details
    {
        "title": "IDEA DETAILS",
        "subtitle": "Proposed Solution",
        "content": """
        <ul class="bullet-list">
            <li><strong>FairFlow</strong> is a "Self-Healing Bias Firewall" that sits between deployed AI models and end-users, ensuring continuous fairness compliance in real-time</li>
            <li>Uses <strong>Deep Reinforcement Learning (PPO)</strong> to dynamically adjust decision thresholds, maintaining an optimal balance between Accuracy (Profit) and Fairness (Compliance)</li>
            <li><strong>Gatekeeper Agent</strong> audits each prediction and decides to <code>APPROVE</code>, <code>DENY</code>, or <code>ESCALATE</code> based on real-time fairness metrics like Demographic Parity and Equalized Odds</li>
            <li>Every RL intervention is logged with <strong>SHAP (Shapley Additive Explanations)</strong> for transparent, explainable decision-making</li>
            <li><strong>Unique Innovation:</strong> Unlike static bias-fixing methods, FairFlow continuously adapts to data drift in production, automatically correcting bias without retraining the base model</li>
        </ul>
        """,
        "slide_num": 3
    },
    
    # Slide 4: Technical Approach
    {
        "title": "TECHNICAL APPROACH",
        "subtitle": "Technologies & Methodology",
        "content": """
        <div class="two-column">
            <div class="column">
                <h3>Technologies Used</h3>
                <table class="tech-table">
                    <tr><th>Layer</th><th>Technology</th></tr>
                    <tr><td>Base ML Model</td><td>XGBoost</td></tr>
                    <tr><td>RL Agent</td><td>Stable-Baselines3 (PPO)</td></tr>
                    <tr><td>RL Environment</td><td>OpenAI Gymnasium</td></tr>
                    <tr><td>Explainability</td><td>SHAP</td></tr>
                    <tr><td>Backend</td><td>FastAPI (Python)</td></tr>
                    <tr><td>Frontend</td><td>Next.js + Recharts</td></tr>
                    <tr><td>Database</td><td>SQLite</td></tr>
                </table>
            </div>
            <div class="column">
                <h3>Step-by-Step Methodology</h3>
                <ol class="method-list">
                    <li>Data Preparation – Load Adult Census Dataset</li>
                    <li>Bias Simulation – Train biased XGBoost model</li>
                    <li>RL Environment – Custom Gym environment</li>
                    <li>RL Training – PPO with composite reward</li>
                    <li>XAI Integration – SHAP explanations</li>
                    <li>Backend – FastAPI endpoints</li>
                    <li>Dashboard – Real-time React interface</li>
                </ol>
            </div>
        </div>
        """,
        "slide_num": 4
    },
    
    # Slide 5: Architecture
    {
        "title": "ARCHITECTURE",
        "subtitle": "Proposed Architecture",
        "content": """
        <div class="architecture-diagram">
            <div class="arch-row">
                <div class="arch-box base-model">
                    <h4>Corporate "Base" Model</h4>
                    <p>XGBoost</p>
                    <p class="small">(Black Box)</p>
                </div>
                <div class="arrow">→</div>
                <div class="arch-box platform">
                    <h4>FairFlow Platform</h4>
                    <div class="platform-inner">
                        <div class="component">RL Gatekeeper<br><span class="small">(PPO Agent)</span></div>
                        <div class="arrow-small">→</div>
                        <div class="component">XAI Engine<br><span class="small">(SHAP)</span></div>
                        <div class="arrow-small">→</div>
                        <div class="component">Audit Log<br><span class="small">(SQLite)</span></div>
                    </div>
                    <div class="decision-bar">Decision: APPROVE / OVERRIDE / ESCALATE</div>
                </div>
            </div>
            <div class="arch-row">
                <div class="arch-box dashboard">
                    <h4>React Dashboard (Next.js)</h4>
                    <ul>
                        <li>Live Accuracy vs. Fairness Charts</li>
                        <li>Intervention History & SHAP Explanations</li>
                        <li>Real-time Bias Monitoring</li>
                    </ul>
                </div>
            </div>
        </div>
        """,
        "slide_num": 5
    },
    
    # Slide 6: Feasibility and Viability
    {
        "title": "FEASIBILITY AND VIABILITY",
        "subtitle": "Feasibility & Challenges",
        "content": """
        <div class="feasibility-section">
            <h3>Feasibility Analysis</h3>
            <ul class="bullet-list small">
                <li><strong>Technical:</strong> Built using established frameworks (Stable-Baselines3, XGBoost, FastAPI, Next.js)</li>
                <li><strong>Data:</strong> Uses publicly available Adult Census Income Dataset (UCI Repository)</li>
                <li><strong>Resource:</strong> Runs on standard hardware; no GPU required for inference</li>
            </ul>
        </div>
        <h3>Challenges & Mitigation</h3>
        <table class="challenge-table">
            <tr><th>Challenge</th><th>Risk</th><th>Mitigation Strategy</th></tr>
            <tr><td>RL Training Instability</td><td>Medium</td><td>Pre-trained agent; rule-based fallback</td></tr>
            <tr><td>SHAP Latency</td><td>Low</td><td>Fast mode; cached explanations</td></tr>
            <tr><td>Real-time Performance</td><td>Medium</td><td>Async processing; optimized pipeline</td></tr>
            <tr><td>Data Drift Handling</td><td>Low</td><td>Continuous monitoring; periodic retraining</td></tr>
        </table>
        """,
        "slide_num": 6
    },
    
    # Slide 7: Impact and Benefits
    {
        "title": "IMPACT AND BENEFITS",
        "subtitle": "Impact & Benefits",
        "content": """
        <div class="two-column">
            <div class="column">
                <h3>Target Audience Impact</h3>
                <ul class="bullet-list small">
                    <li><strong>Banks:</strong> Compliant loan/credit decisions with EU AI Act & GDPR</li>
                    <li><strong>HR Firms:</strong> Fair hiring algorithms across demographics</li>
                    <li><strong>Insurance:</strong> Equitable premium/claim decisions</li>
                    <li><strong>Compliance Officers:</strong> Real-time dashboard & audit trail</li>
                </ul>
            </div>
            <div class="column">
                <h3>Key Benefits</h3>
                <table class="benefit-table">
                    <tr><td class="benefit-type">Regulatory</td><td>EU AI Act Article 9 compliance</td></tr>
                    <tr><td class="benefit-type">Economic</td><td>Reduces legal risk & retraining costs</td></tr>
                    <tr><td class="benefit-type">Social</td><td>Equitable AI across all groups</td></tr>
                    <tr><td class="benefit-type">Operational</td><td>Self-healing, continuous monitoring</td></tr>
                    <tr><td class="benefit-type">Transparency</td><td>SHAP explanations for audit trail</td></tr>
                </table>
            </div>
        </div>
        """,
        "slide_num": 7
    },
    
    # Slide 8: Comparison
    {
        "title": "COMPARISON WITH EXISTING SYSTEM",
        "subtitle": "Comparison",
        "content": """
        <table class="comparison-table">
            <tr>
                <th>Feature</th>
                <th>Traditional Bias Mitigation</th>
                <th>FairFlow (Our Solution)</th>
            </tr>
            <tr><td>Approach</td><td>Static, one-time fix</td><td class="highlight">Dynamic, real-time adaptation</td></tr>
            <tr><td>Data Drift</td><td>Requires model retrain</td><td class="highlight">Auto-corrects via RL</td></tr>
            <tr><td>Explainability</td><td>Limited or none</td><td class="highlight">Full SHAP explanations</td></tr>
            <tr><td>Audit Trail</td><td>Manual logging</td><td class="highlight">Automatic, immutable log</td></tr>
            <tr><td>Integration</td><td>Requires model access</td><td class="highlight">Model-agnostic wrapper</td></tr>
            <tr><td>Compliance</td><td>Periodic manual audits</td><td class="highlight">Continuous monitoring</td></tr>
            <tr><td>Human-in-Loop</td><td>Not supported</td><td class="highlight">Escalate action available</td></tr>
            <tr><td>Deployment</td><td>Replace entire model</td><td class="highlight">Plug-and-play middleware</td></tr>
        </table>
        """,
        "slide_num": 8
    }
]

# CSS Styles for the presentation
CSS_STYLES = """
@page {
    size: 1280px 720px;
    margin: 0;
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: #1a1a2e;
    color: #ffffff;
}

.slide {
    width: 1280px;
    height: 720px;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 40px 60px;
    page-break-after: always;
    position: relative;
    overflow: hidden;
}

.slide::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 6px;
    background: linear-gradient(90deg, #e94560, #ff6b6b, #ffd93d, #6bcb77, #4d96ff);
}

.header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 30px;
    padding-bottom: 20px;
    border-bottom: 2px solid rgba(255,255,255,0.1);
}

.header h1 {
    font-size: 36px;
    font-weight: 700;
    color: #e94560;
    text-transform: uppercase;
    letter-spacing: 2px;
}

.logo-placeholder {
    font-size: 14px;
    color: rgba(255,255,255,0.6);
    text-align: right;
}

.subtitle {
    font-size: 24px;
    color: #ffd93d;
    margin-bottom: 25px;
    font-weight: 500;
}

.content {
    flex: 1;
}

/* Form Grid for Slide 1 */
.form-grid {
    background: rgba(255,255,255,0.05);
    border-radius: 15px;
    padding: 25px 35px;
    margin-bottom: 30px;
}

.form-row {
    display: flex;
    padding: 12px 0;
    border-bottom: 1px solid rgba(255,255,255,0.1);
}

.form-row:last-child {
    border-bottom: none;
}

.label {
    font-weight: 600;
    color: #4d96ff;
    width: 200px;
    font-size: 18px;
}

.value {
    color: #ffffff;
    font-size: 18px;
}

.project-title {
    text-align: center;
    margin-top: 30px;
}

.project-title h2 {
    font-size: 32px;
    color: #6bcb77;
    margin-bottom: 10px;
}

.tagline {
    font-size: 18px;
    color: rgba(255,255,255,0.7);
    font-style: italic;
}

/* Team Grid */
.team-grid {
    display: flex;
    justify-content: space-between;
    gap: 20px;
    margin-top: 20px;
}

.team-member {
    background: rgba(255,255,255,0.05);
    border-radius: 15px;
    padding: 25px 15px;
    text-align: center;
    flex: 1;
    border: 1px solid rgba(255,255,255,0.1);
}

.avatar {
    font-size: 48px;
    margin-bottom: 15px;
}

.team-member h3 {
    font-size: 16px;
    color: #ffffff;
    margin-bottom: 8px;
}

.role {
    color: #e94560;
    font-weight: 600;
    font-size: 14px;
    margin-bottom: 10px;
}

.details {
    color: rgba(255,255,255,0.6);
    font-size: 12px;
    line-height: 1.4;
}

/* Bullet Lists */
.bullet-list {
    list-style: none;
    padding-left: 0;
}

.bullet-list li {
    padding: 15px 20px;
    margin-bottom: 12px;
    background: rgba(255,255,255,0.05);
    border-radius: 10px;
    border-left: 4px solid #4d96ff;
    font-size: 17px;
    line-height: 1.5;
}

.bullet-list.small li {
    padding: 10px 15px;
    margin-bottom: 8px;
    font-size: 15px;
}

code {
    background: rgba(233, 69, 96, 0.3);
    padding: 2px 8px;
    border-radius: 4px;
    font-family: 'Consolas', monospace;
    color: #ff6b6b;
}

/* Two Column Layout */
.two-column {
    display: flex;
    gap: 40px;
}

.column {
    flex: 1;
}

.column h3 {
    color: #ffd93d;
    font-size: 20px;
    margin-bottom: 15px;
    padding-bottom: 10px;
    border-bottom: 2px solid rgba(255,255,255,0.1);
}

/* Tables */
table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 10px;
}

th, td {
    padding: 12px 15px;
    text-align: left;
    border-bottom: 1px solid rgba(255,255,255,0.1);
}

th {
    background: rgba(77, 150, 255, 0.2);
    color: #4d96ff;
    font-weight: 600;
    font-size: 14px;
}

td {
    font-size: 14px;
}

.tech-table td:first-child {
    color: rgba(255,255,255,0.7);
}

.tech-table td:last-child {
    color: #6bcb77;
    font-weight: 500;
}

.method-list {
    padding-left: 20px;
}

.method-list li {
    padding: 8px 0;
    font-size: 14px;
    color: rgba(255,255,255,0.9);
}

/* Architecture Diagram */
.architecture-diagram {
    margin-top: 20px;
}

.arch-row {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 30px;
    margin-bottom: 30px;
}

.arch-box {
    background: rgba(255,255,255,0.05);
    border-radius: 15px;
    padding: 25px;
    border: 2px solid rgba(255,255,255,0.1);
}

.base-model {
    text-align: center;
    min-width: 180px;
    border-color: #e94560;
}

.platform {
    flex: 1;
    max-width: 700px;
    border-color: #4d96ff;
}

.platform h4 {
    color: #4d96ff;
    margin-bottom: 15px;
    text-align: center;
}

.platform-inner {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 15px;
    margin-bottom: 15px;
}

.component {
    background: rgba(77, 150, 255, 0.2);
    padding: 15px 20px;
    border-radius: 10px;
    text-align: center;
    font-size: 14px;
}

.arrow, .arrow-small {
    color: #ffd93d;
    font-size: 24px;
    font-weight: bold;
}

.arrow-small {
    font-size: 18px;
}

.decision-bar {
    background: rgba(233, 69, 96, 0.2);
    padding: 10px;
    border-radius: 8px;
    text-align: center;
    color: #e94560;
    font-weight: 600;
    font-size: 14px;
}

.dashboard {
    width: 100%;
    max-width: 500px;
    margin: 0 auto;
    border-color: #6bcb77;
}

.dashboard h4 {
    color: #6bcb77;
    margin-bottom: 10px;
}

.dashboard ul {
    padding-left: 20px;
    font-size: 14px;
}

.dashboard li {
    padding: 5px 0;
    color: rgba(255,255,255,0.8);
}

.arch-box h4 {
    color: #e94560;
    margin-bottom: 5px;
    font-size: 16px;
}

.arch-box p {
    font-size: 14px;
    color: rgba(255,255,255,0.8);
}

.small {
    font-size: 12px !important;
    color: rgba(255,255,255,0.6) !important;
}

/* Feasibility Section */
.feasibility-section {
    margin-bottom: 25px;
}

.feasibility-section h3 {
    color: #6bcb77;
    margin-bottom: 15px;
    font-size: 20px;
}

h3 {
    color: #ffd93d;
    font-size: 18px;
    margin-bottom: 15px;
}

.challenge-table td:first-child {
    font-weight: 500;
}

.challenge-table td:nth-child(2) {
    color: #ffd93d;
}

/* Benefits */
.benefit-table td {
    padding: 10px 12px;
}

.benefit-type {
    color: #4d96ff;
    font-weight: 600;
    width: 100px;
}

/* Comparison Table */
.comparison-table {
    font-size: 13px;
}

.comparison-table th {
    font-size: 14px;
}

.comparison-table td.highlight {
    color: #6bcb77;
    font-weight: 500;
}

/* Footer */
.footer {
    position: absolute;
    bottom: 20px;
    right: 40px;
    font-size: 18px;
    color: rgba(255,255,255,0.4);
}

/* ImpactThon Logo area */
.impactthon-badge {
    position: absolute;
    top: 20px;
    right: 40px;
    text-align: right;
    font-size: 12px;
    color: rgba(255,255,255,0.5);
}

.impactthon-badge .title {
    font-size: 14px;
    color: #ffd93d;
    font-weight: 600;
}
"""

def generate_slide_html(slide):
    """Generate HTML for a single slide"""
    subtitle_html = f'<h2 class="subtitle">{slide.get("subtitle", "")}</h2>' if slide.get("subtitle") else ""
    
    return f"""
    <div class="slide">
        <div class="impactthon-badge">
            <div class="title">ImpactThon @KSV</div>
            <div>2025 - 2026</div>
        </div>
        <div class="header">
            <h1>{slide['title']}</h1>
            <div class="logo-placeholder">KSV | IEEE Student Branch</div>
        </div>
        {subtitle_html}
        <div class="content">
            {slide['content']}
        </div>
        <div class="footer">{slide['slide_num']}</div>
    </div>
    """

def generate_full_html():
    """Generate the complete HTML document"""
    slides_html = "\n".join(generate_slide_html(slide) for slide in SLIDES)
    
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=1280, height=720">
    <title>FairFlow - ImpactThon Presentation</title>
    <style>
        {CSS_STYLES}
    </style>
</head>
<body>
    {slides_html}
</body>
</html>
"""

def save_html(output_path):
    """Save the HTML file"""
    html_content = generate_full_html()
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"✅ HTML saved to: {output_path}")
    return output_path

def convert_to_pdf_playwright(html_path, pdf_path):
    """Convert HTML to PDF using Playwright"""
    try:
        from playwright.sync_api import sync_playwright
        
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            
            # Load the HTML file
            page.goto(f'file:///{html_path.replace(os.sep, "/")}')
            page.wait_for_load_state('networkidle')
            
            # Generate PDF with slide dimensions
            page.pdf(
                path=pdf_path,
                width='1280px',
                height='720px',
                print_background=True,
                margin={'top': '0', 'right': '0', 'bottom': '0', 'left': '0'}
            )
            
            browser.close()
            print(f"✅ PDF saved to: {pdf_path}")
            return True
    except ImportError:
        print("❌ Playwright not installed. Install with: pip install playwright && playwright install chromium")
        return False
    except Exception as e:
        print(f"❌ Playwright error: {e}")
        return False

def convert_to_pdf_weasyprint(html_path, pdf_path):
    """Convert HTML to PDF using WeasyPrint"""
    try:
        from weasyprint import HTML, CSS
        
        HTML(filename=html_path).write_pdf(pdf_path)
        print(f"✅ PDF saved to: {pdf_path}")
        return True
    except ImportError:
        print("❌ WeasyPrint not installed. Install with: pip install weasyprint")
        return False
    except Exception as e:
        print(f"❌ WeasyPrint error: {e}")
        return False

def main():
    # Get the directory of this script
    script_dir = Path(__file__).parent
    
    # Output paths
    html_path = script_dir / "presentation.html"
    pdf_path = script_dir / "FairFlow_Presentation.pdf"
    
    # Generate HTML
    save_html(str(html_path))
    
    print("\n📄 Attempting PDF generation...")
    
    # Try Playwright first (better rendering)
    if convert_to_pdf_playwright(str(html_path), str(pdf_path)):
        print("\n🎉 Presentation generated successfully!")
        print(f"   HTML: {html_path}")
        print(f"   PDF:  {pdf_path}")
        return
    
    # Fallback to WeasyPrint
    print("\n🔄 Trying WeasyPrint as fallback...")
    if convert_to_pdf_weasyprint(str(html_path), str(pdf_path)):
        print("\n🎉 Presentation generated successfully!")
        print(f"   HTML: {html_path}")
        print(f"   PDF:  {pdf_path}")
        return
    
    # Manual instructions if both fail
    print("\n" + "="*60)
    print("📋 MANUAL PDF GENERATION INSTRUCTIONS:")
    print("="*60)
    print(f"1. Open the HTML file: {html_path}")
    print("2. Open it in Chrome/Edge browser")
    print("3. Press Ctrl+P (Print)")
    print("4. Select 'Save as PDF' as destination")
    print("5. Set Paper size to 'Custom' (1280 x 720 px) or Landscape")
    print("6. Disable margins (set to None)")
    print("7. Enable 'Background graphics'")
    print("8. Save as PDF")
    print("="*60)

if __name__ == "__main__":
    main()
