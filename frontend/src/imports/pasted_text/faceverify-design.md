Design a polished, modern LIGHT-THEME desktop web application for a hackathon project called:

“FaceVerify”
Face Identification & Blockchain Verification

This is the frontend UI for an existing Python backend pipeline.

IMPORTANT:
The backend already exists and works. The UI should be designed as a clean presentation and control layer around the existing pipeline.

CORE PIPELINE:

Face Image
→ Face Detection
→ Face Embedding
→ Google Lens / SerpApi Web Search
→ Candidate Images
→ Independent Face Similarity Verification
→ SHA-256 Fingerprint
→ Blockchain Record
→ Chain Integrity Verification
→ On-Chain Re-verification

The application is a technical verification tool.

Do NOT present it as an identity-surveillance system.
Do NOT use wording such as “100% identity confirmed”.
Use:
- Face Similarity
- Verification Signal
- Candidate Result
- Verification Status
- Source
- Blockchain Verification

--------------------------------------------------
VISUAL DIRECTION
--------------------------------------------------

Use a PREMIUM LIGHT THEME.

The design should feel like a combination of:

- modern developer SaaS
- cybersecurity product
- AI verification platform
- premium fintech dashboard

Avoid the typical “AI = dark neon dashboard” style.

The interface should feel:
- clean
- trustworthy
- technical
- sophisticated
- minimal
- professional
- spacious
- easy to understand

PRIMARY BACKGROUND:

Warm off-white / very light cool gray.

Suggested direction:
#F7F8FA or similar.

CARDS:

Pure white:
#FFFFFF

Use very subtle borders:
#E5E7EB

Use soft shadows, not heavy shadows.

TEXT:

Primary:
Deep navy / charcoal

Suggested:
#111827

Secondary:
#667085

Muted:
#98A2B3

PRIMARY BUTTONS:

Use DARK NAVY / CHARCOAL buttons.

Example:
#111827

White text.

Buttons should have:
- 10–12px rounded corners
- subtle hover state
- strong contrast
- professional appearance

SUCCESS:

Use a sophisticated emerald/teal.

Example direction:
#0F9D78

Use it for:
- VERIFIED
- PASSED
- Face detected
- Blockchain verified
- successful pipeline steps

INFORMATION:

Use muted blue.

Example:
#2563EB

WARNING:

Use amber:
#D97706

ERROR:

Use muted red:
#DC2626

Do not overuse colors.
Color should communicate status rather than decoration.

--------------------------------------------------
TYPOGRAPHY
--------------------------------------------------

Use a modern professional sans-serif such as:

Inter
or
Manrope

Headings:
Bold / semibold

Body:
Regular

Technical values such as:
- hashes
- URLs
- similarity scores
- blockchain IDs
- timestamps

should use a monospace font such as:
JetBrains Mono

--------------------------------------------------
OVERALL LAYOUT
--------------------------------------------------

Desktop-first design.

Primary frame:
1440 × 1024.

Use a centered application container with generous horizontal spacing.

Structure:

TOP NAVIGATION
↓
PAGE HEADER
↓
MAIN VERIFICATION WORKSPACE
↓
PIPELINE STATUS
↓
RESULT + BLOCKCHAIN DETAILS
↓
PIPELINE LOG

Use a clean 12-column grid.

Do not make the UI feel crowded.

--------------------------------------------------
1. TOP NAVIGATION
--------------------------------------------------

Create a clean white navigation bar.

Left:

Small minimal logo combining:
- face outline
- verification/checkmark

Product name:

FaceVerify

Small badge beside it:

HH Goa 2026 · Task 3

Navigation:

Verification
History
System

Right side:

Small green status dot

“System Ready”

User/avatar icon can be included as a purely visual element, but do not create unnecessary user-account functionality.

Navbar should have a subtle bottom border.

--------------------------------------------------
2. PAGE HEADER
--------------------------------------------------

Large heading:

“Face Verification Pipeline”

Subtitle:

“Discover visual candidates on the open web, independently verify the face, and anchor verified results to a tamper-evident ledger.”

Add a small status pill:

● Pipeline Ready

On the right side, optionally show:

“Local Verification Engine”

with a small shield/check icon.

--------------------------------------------------
3. MAIN WORKSPACE
--------------------------------------------------

Create a large two-column layout.

LEFT:
Face Input

RIGHT:
Verification Overview

--------------------------------------------------
4. FACE INPUT CARD
--------------------------------------------------

Card title:

“01 · Face Input”

Subtitle:

“Upload an image containing a detectable face.”

Large drag-and-drop area.

Inside:

Minimal upload icon.

Text:

“Drop an image here”

Secondary:

“or browse from your computer”

Below:

“JPG, JPEG or PNG · Max 10 MB”

Use a subtle dashed border.

After upload, transform the area into an image preview.

Display:
- uploaded image
- face bounding box
- small label:

“Face detected”

Show:

Detection confidence
0.8149

Embedding:

512D generated

Use a small green verification indicator.

Primary dark button:

“Run Verification”

Secondary outline button:

“Clear”

--------------------------------------------------
5. VERIFICATION OVERVIEW CARD
--------------------------------------------------

Title:

“Verification Overview”

Show a clean vertical status timeline:

Face Detection
✓ Complete

Face Embedding
✓ Complete

Web Search
✓ Complete

Candidate Verification
✓ Complete

Blockchain Registration
✓ Complete

Re-verification
✓ Complete

Use a thin connecting vertical line.

Completed steps use subtle emerald indicators.

Do not use excessive icons.

--------------------------------------------------
6. PIPELINE VISUALIZATION
--------------------------------------------------

Below the two cards, create a horizontal pipeline.

Display:

FACE
↓
SEARCH
↓
VERIFY
↓
BLOCKCHAIN
↓
RE-VERIFY

Use elegant numbered circles:

01
02
03
04
05

Each completed step has a subtle green indicator.

The current/active step should use blue.

The pipeline should immediately communicate the application's purpose.

--------------------------------------------------
7. WEB SEARCH RESULTS
--------------------------------------------------

Large section title:

“02 · Web Search Results”

Top-right:

“59 candidates discovered”

Add a small explanatory label:

“Candidates retrieved dynamically from Google Lens”

Create a prominent candidate result card.

CARD:

Left:
Candidate image thumbnail.

Right:

Facebook

Source page title:

“Photo of Albert Einstein was taken by photographer…”

Source:

facebook.com

Add:

“Open Source ↗”

Then show a verification section.

Face Similarity

0.9709

Threshold

0.5000

Status:

VERIFIED

Make 0.9709 visually prominent.

Use a subtle emerald accent around the VERIFIED status.

Below it:

“Candidate face exceeds the configured similarity threshold.”

IMPORTANT:
Do NOT say:
“Identity confirmed.”

--------------------------------------------------
8. CANDIDATE METRICS
--------------------------------------------------

Under the primary result, create compact metric cards:

Candidates Evaluated
59

Faces Detected
1+

Best Similarity
0.9709

Threshold
0.5000

Keep these cards minimal.

--------------------------------------------------
9. BLOCKCHAIN VERIFICATION
--------------------------------------------------

Create a large premium white card.

Title:

“03 · Blockchain Verification”

At top-right:

✓ VERIFIED

Use an emerald status pill.

Display:

SHA-256 Fingerprint

2a1929b44f3d4eb3b541db508f9335491444515a14825348e87edfc82ca30550

Add copy icon.

Then:

Blockchain Record

46e594d4afb39a7a53c7d5f605a9f23dca111318f3adcf42b81f9484aa721fe5

Add copy icon.

Then create three status rows:

Chain Integrity
PASSED ✓

Record Status
REGISTERED ✓

On-chain Re-verification
VERIFIED · 100%

Use monospace typography for hashes.

--------------------------------------------------
10. BLOCKCHAIN CHAIN VISUAL
--------------------------------------------------

Create a minimal horizontal chain visualization:

Genesis
   ↓
Block
   ↓
Verified Record
   ↓
Integrity Check

Use small connected blocks.

Avoid making it look like cryptocurrency trading.

The visual purpose is:
tamper-evident record verification.

--------------------------------------------------
11. VERIFICATION SUMMARY
--------------------------------------------------

Create a compact summary card:

“Verification Summary”

Rows:

Input
einstein_demo.jpg

Candidates
59

Source
Facebook

Similarity
0.9709

Threshold
0.5000

Blockchain
Registered

Integrity
PASSED

Re-verification
VERIFIED

Use a clean two-column layout.

--------------------------------------------------
12. PIPELINE LOG
--------------------------------------------------

At the bottom create:

“Pipeline Activity”

A white card containing a light terminal-style log.

Use very light gray background inside the log.

Monospace font.

Example:

[14:32:01] Loading input image...
[14:32:02] Face detected
[14:32:02] Generating 512D embedding...
[14:32:03] Uploading face crop to Google Lens...
[14:32:05] 59 candidates retrieved
[14:32:07] Candidate verification started...
[14:32:09] Candidate similarity: 0.9709
[14:32:09] Verification threshold: 0.5000
[14:32:09] Candidate VERIFIED
[14:32:10] SHA-256 fingerprint generated
[14:32:10] Blockchain record created
[14:32:10] Chain integrity PASSED
[14:32:10] Re-verification VERIFIED

Use small colored status dots.

--------------------------------------------------
13. FAILURE STATE
--------------------------------------------------

Design a second state for:

“NO VERIFIED MATCH FOUND”

Example:

59 candidates evaluated

Highest similarity
0.3699

Required threshold
0.5000

Blockchain
No record created

Message:

“No candidate exceeded the configured verification threshold. No blockchain record was created.”

Use amber/red status indicators sparingly.

The failure state should look professional rather than alarming.

--------------------------------------------------
14. LOADING STATE
--------------------------------------------------

Design a processing state.

After clicking:

“Run Verification”

Show:

Detecting face...
Generating embedding...
Searching Google Lens...
Evaluating candidates...
Registering blockchain record...
Re-verifying record...

Use animated-looking skeleton/progress indicators.

Do NOT display fake percentages.

Use:
“Processing”
“Searching”
“Verifying”

instead.

--------------------------------------------------
15. HISTORY PAGE
--------------------------------------------------

Create a clean history page.

Title:

“Verification History”

Table:

Date
Input
Candidates
Similarity
Source
Blockchain
Status

Example:

Sep 5, 2026
einstein_demo.jpg
59
0.9709
Facebook
Verified
Success

IMPORTANT:
This is only a UI representation.
Do not imply that persistent history storage currently exists in the backend.

--------------------------------------------------
16. SYSTEM PAGE
--------------------------------------------------

Create:

“System Configuration”

Cards:

Face Engine
InsightFace · buffalo_l

Embedding
512 dimensions

Search Provider
Google Lens via SerpApi

Verification Method
L2-normalized cosine similarity

Threshold
0.5000

Fingerprint
SHA-256

Blockchain
Local SHA-256 hash-chain ledger

Re-verification
Enabled

Add an informational note:

“API credentials are handled server-side and are never displayed in the interface.”

--------------------------------------------------
17. RESPONSIVE DESIGN
--------------------------------------------------

Create responsive behavior for:

Desktop
Tablet
Mobile

On mobile:
- two-column layout becomes one column
- pipeline becomes vertical
- hashes wrap
- candidate card becomes stacked
- buttons become full-width where appropriate

--------------------------------------------------
18. MICRO-INTERACTIONS
--------------------------------------------------

Use subtle professional interactions:

- button hover
- card hover
- upload drag state
- verification success animation
- checkmark transition
- pipeline step completion
- copy-to-clipboard feedback

Keep animations fast and subtle.

Do NOT use flashy neon effects.

--------------------------------------------------
19. DESIGN SYSTEM
--------------------------------------------------

Create reusable components:

Navbar
Button
Badge
StatusBadge
UploadDropzone
ImagePreview
FaceDetectionStatus
PipelineStepper
CandidateCard
SimilarityScore
MetricCard
BlockchainCard
HashField
ChainVisualizer
PipelineLog
SummaryCard
LoadingState
SuccessState
FailureState
DataTable

Create consistent:
- spacing
- border radius
- typography
- button styles
- status colors
- card styles

--------------------------------------------------
20. FINAL VISUAL GOAL
--------------------------------------------------

The final product should look like a premium technical verification platform.

Think:

“Modern AI security SaaS + developer tool + fintech-grade audit interface.”

NOT:

- generic AI dashboard
- dark cyberpunk interface
- crypto trading dashboard
- social media clone
- surveillance system

The first screen should communicate the complete concept immediately:

UPLOAD FACE
↓
SEARCH WEB
↓
VERIFY CANDIDATE
↓
ANCHOR TO BLOCKCHAIN
↓
RE-VERIFY

The UI must prioritize clarity, technical credibility, and evidence of the working pipeline over decorative elements.

Use a LIGHT BODY with WHITE CARDS and DARK NAVY BUTTONS as the primary visual foundation.