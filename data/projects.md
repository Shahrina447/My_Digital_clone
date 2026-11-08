# Building Practical AI Projects: DeadlineOS and Verdade

## About this document

I am Shahrina Khan, a computer science student at UMT Sialkot. My GitHub profile describes me as a developer and researcher interested in building projects at the intersection of AI and real-world problems. Two projects that show this interest are DeadlineOS and Verdade.

They address different needs. DeadlineOS is an AI-assisted planning and workload-management platform. Verdade is an Urdu-language fake-news detection project. One focuses on organizing work before problems happen; the other explores how natural-language processing can help analyze news claims in a language that is often underserved by digital tools.

This write-up describes the goals, designs, technologies, and limitations documented for both projects.

## DeadlineOS: proactive planning

### The problem

Many task-management tools are primarily reactive. They let a person record tasks and deadlines, but may not help them notice that their workload is becoming unrealistic until a deadline is close or has already been missed.

DeadlineOS is designed around the idea of proactive planning. Its project description calls the problem “schedule density blindness”: a person may have many commitments without a clear picture of how those commitments interact or whether there is enough time to complete them.

The aim is to bring tasks, goals, habits, meetings, and deadlines into a coordinated system. Rather than serving only as a list, DeadlineOS is intended to help users understand their workload and identify potential conflicts earlier.

### What the system is intended to do

DeadlineOS combines planning features with several types of AI-assisted input. Its README describes voice, vision, and document workflows:

- Voice input can be used to express tasks or request actions in natural language.
- Vision features can extract task information from images, screenshots, or whiteboards.
- Document intelligence can turn information in PDFs or DOCX files into tracked milestones.
- A planner can place tasks into available calendar time while considering workload limits.
- A digital-twin feature can simulate how likely a user is to complete upcoming goals.
- A Rescue Center can identify tasks that may be at risk and suggest recovery steps.

These features follow a common design goal: reduce the distance between noticing a problem and taking action. A user should be able to see a task, understand how it affects the schedule, and receive help deciding what to do next.

### Architecture and technology

The README describes DeadlineOS as a web application with a React and TypeScript frontend and a Python backend. The frontend uses Vite, Tailwind CSS, and Framer Motion. The backend uses Flask and SQLAlchemy, with a serverless PostgreSQL database hosted on Neon.

Authentication is described as using Supabase Auth and JWT verification. The system can use Google Gemini 2.0 Flash for more complex AI work, while a local intelligence engine handles simpler natural-language processing tasks. The README describes this as a hybrid inference approach: handle straightforward requests locally, and use the external model when local confidence is low or the request is ambiguous.

This split is useful because not every interaction needs the same amount of computation. A simple intent or entity extraction task may not need a large model call. More complicated requests may benefit from a generative model. The intended result is a balance between speed, privacy, and the ability to handle complex requests.

### Designing for risk, not just task lists

A central idea in DeadlineOS is to treat planning as a question of capacity. A list can show what needs to be done, but a schedule can also show when the work might fit. If tasks compete for the same time, a planner can help make that conflict visible.

The project’s digital-twin concept extends this idea. It is intended to compare a user’s past completion velocity with future workload and simulate possible completion trajectories. Such a feature should be understood as decision support, not a guarantee. A simulation can help expose risk, but it cannot know every future interruption or change in circumstances.

The Rescue Center takes a related approach. Instead of merely flagging an at-risk task, it is intended to suggest a multi-step recovery strategy. This reflects a practical principle: identifying a problem is more useful when the system also helps the user consider a next step.

### Implementation and deployment

The project README describes a frontend deployed through Vercel and a backend deployed through Render, with Neon PostgreSQL and Supabase providing supporting services. The repository is organized into frontend and backend areas, with backend folders for API boundaries, database configuration, models, services, and utilities.

The README also describes measures such as rate limiting, input validation, environment variables for secrets, and tenant isolation. These are important because a productivity application may handle personal schedules and user data. Security and privacy need to be considered alongside the visible planning features.

## Verdade: Urdu-language news classification

Verdade is an AI-powered Urdu fake-news detection platform. Its README describes the project as an academic NLP project intended for course demonstration. It accepts Urdu news text and uses a fine-tuned XLM-RoBERTa model to produce a classification and confidence information.

The motivation is linguistic access. The project README notes that misinformation circulates through Urdu-language social media, while many existing fact-checking tools focus on English or depend on manual review. Verdade explores how a multilingual transformer can be adapted to a specific language and task.

### How Verdade processes an article

The documented prediction flow is:

1. A user pastes Urdu news text into the application.
2. The text is tokenized with the XLM-RoBERTa tokenizer, with a maximum length of 256 tokens.
3. The fine-tuned model performs an inference pass.
4. The system calculates confidence values for the output classes.
5. A post-processing layer can return an uncertain result or apply documented heuristic patterns.
6. The backend returns a structured response, and the frontend displays the result.
7. Prediction history can be stored with an identifier and timestamp.

The output includes a prediction, confidence values, verdict text, and a prediction ID. The interface is described as supporting right-to-left Urdu rendering, which matters for presenting Urdu text naturally and legibly.

### Dataset and evaluation

The README describes a Combined Urdu News Dataset with 10,421 samples: 3,339 in the fake class and 7,082 in the true class. It documents several evaluation sets, including an in-distribution test set, an external test set, and synthetic samples.

The README reports 98% accuracy on the in-distribution set and 96.25% on an external set. It reports 88.75% accuracy on the synthetic set. These are results reported by the project, not a promise that every future prediction will be correct. Model performance depends on the data, the evaluation design, and how closely future articles resemble the examples used during training and testing.

The different test sets are important because a single score can give an incomplete picture. An in-distribution test measures performance on held-out examples from a related dataset. An external test can provide evidence about whether a model generalizes to a different source. Synthetic examples can expose behavior on generated or hand-crafted text, though they may not represent the full range of real-world news.

### Handling uncertainty

Verdade documents a confidence threshold: when the model’s highest confidence is below 65%, it returns `UNCERTAIN` rather than forcing a real-or-fake label. This is a valuable design choice for a tool dealing with information that could influence what people believe or share.

The project also documents heuristic overrides for certain Urdu patterns, including conspiracy framing, miracle-cure claims, and suppression or cover-up language. These rules are intended to catch patterns that the model may miss. Heuristics can be useful, but they also require careful testing: a phrase that appears in misinformation may also appear in legitimate reporting that discusses or debunks misinformation.

For these reasons, Verdade should be treated as an aid for further checking, not as an authoritative source of truth. Important claims should still be verified through trusted sources.

### What the projects have in common

DeadlineOS and Verdade solve different problems, but they share several design themes:

- **Practical intent:** Each project is built around a recognizable real-world need.
- **Structured systems:** Both combine interfaces, backend services, and data-processing logic.
- **AI with supporting logic:** Neither project is only a model call; each includes application behavior around the model or intelligence layer.
- **Attention to the user experience:** DeadlineOS aims to make workload and next steps visible; Verdade includes structured verdicts and Urdu RTL support.
- **The importance of uncertainty:** DeadlineOS simulations should be interpreted as estimates, and Verdade can return `UNCERTAIN` when confidence is low.

A useful AI application needs more than an impressive model. It also needs a clear purpose, understandable outputs, appropriate safeguards, and honest communication about limitations.

## Closing reflection

These projects represent two ways of applying computer science to everyday challenges. DeadlineOS explores how software can help people coordinate tasks and respond to workload risk. Verdade explores how NLP can be adapted to Urdu news analysis while making uncertainty visible.

Both projects are examples of learning through building. They bring together frontend development, backend systems, data handling, and AI. They also show why responsible product design matters: predictions and recommendations should help people make better-informed decisions, not replace judgment or verification.

## Project references

- DeadlineOS: https://github.com/Shahrina447/FOCUSOS
- Verdade: https://github.com/Shahrina447/Verdade
- Urdu news dataset: https://github.com/Shahrina447/urdu-news-dataset
- Profile: https://github.com/Shahrina447
