# Deployment Flow Diagram

This diagram shows the complete deployment pipeline for ML models in the MLM-101 course.

## Mermaid Diagram (renders in GitHub)

```mermaid
flowchart TB
    subgraph Dev["💻 Development Environment"]
        Notebook[Jupyter Notebook<br/>Model Development]
        Train[Train Model<br/>Evaluate Performance]
        Save[Save Model<br/>joblib/pickle/h5]
    end

    subgraph Serial["📦 Model Serialization"]
        Export[Export Model]
        Format{Model<br/>Format}
        Joblib[.joblib<br/>scikit-learn]
        H5[.h5/.keras<br/>TensorFlow]
        Pickle[.pkl<br/>Python]
    end

    subgraph API["🔌 API Layer"]
        FastAPI[FastAPI Server<br/>REST Endpoints]
        Routes[Define Routes<br/>/predict, /health]
        Pydantic[Request Validation<br/>Pydantic Models]
    end

    subgraph UI["🖥️ User Interface"]
        Streamlit[Streamlit App<br/>Interactive UI]
        Gradio[Gradio Interface<br/>Quick Demo]
        HTML[Custom HTML/JS<br/>Frontend]
    end

    subgraph Container["🐳 Containerization"]
        Dockerfile[Create Dockerfile<br/>Define Environment]
        Build[Build Image<br/>docker build]
        Registry[Push to Registry<br/>Docker Hub/ECR]
    end

    subgraph Deploy["☁️ Deployment"]
        Local[Local Server<br/>Development]
        Cloud[Cloud Platform]
        StreamlitCloud[Streamlit Cloud]
        Heroku[Heroku]
        AWS[AWS EC2/ECS]
        Azure[Azure App Service]
    end

    subgraph Monitor["📊 Monitoring"]
        Logs[Application Logs]
        Metrics[Performance Metrics]
        Alerts[Alert System]
    end

    Notebook --> Train
    Train --> Save
    Save --> Export
    Export --> Format

    Format --> Joblib
    Format --> H5
    Format --> Pickle

    Joblib --> FastAPI
    H5 --> FastAPI
    Pickle --> FastAPI

    Joblib --> Streamlit
    H5 --> Streamlit

    FastAPI --> Routes
    Routes --> Pydantic

    Streamlit --> Dockerfile
    FastAPI --> Dockerfile
    Gradio --> Dockerfile

    Dockerfile --> Build
    Build --> Registry

    Registry --> Local
    Registry --> Cloud

    Cloud --> StreamlitCloud
    Cloud --> Heroku
    Cloud --> AWS
    Cloud --> Azure

    Local --> Monitor
    StreamlitCloud --> Monitor
    Heroku --> Monitor
    AWS --> Monitor
    Azure --> Monitor

    Monitor --> Logs
    Monitor --> Metrics
    Monitor --> Alerts

    style Dev fill:#E8F5E9
    style Serial fill:#E3F2FD
    style API fill:#FFF9C4
    style UI fill:#FCE4EC
    style Container fill:#E1F5FE
    style Deploy fill:#F3E5F5
    style Monitor fill:#FFF3E0
```

## Detailed Deployment Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                    DEPLOYMENT FLOW                               │
└──────────────────────────────────────────────────────────────────┘

STAGE 1: MODEL DEVELOPMENT
┌─────────────────────────────────────────────┐
│  Jupyter Notebook                           │
│  ┌───────────────────────────────┐          │
│  │ 1. Load Data                  │          │
│  │ 2. Preprocess                 │          │
│  │ 3. Train Model                │          │
│  │ 4. Evaluate (R², Accuracy)    │          │
│  │ 5. Tune Hyperparameters       │          │
│  └───────────────┬───────────────┘          │
└──────────────────┼──────────────────────────┘
                   │
                   ▼
STAGE 2: MODEL SERIALIZATION
┌─────────────────────────────────────────────┐
│  Save Trained Model                         │
│                                             │
│  Option A: scikit-learn                    │
│  ┌─────────────────────────────────┐       │
│  │ import joblib                   │       │
│  │ joblib.dump(model, 'model.pkl') │       │
│  └─────────────────────────────────┘       │
│                                             │
│  Option B: TensorFlow/Keras                │
│  ┌─────────────────────────────────┐       │
│  │ model.save('model.h5')          │       │
│  └─────────────────────────────────┘       │
│                                             │
│  Option C: ONNX (Universal)                │
│  ┌─────────────────────────────────┐       │
│  │ import onnx                     │       │
│  │ onnx.save(model, 'model.onnx')  │       │
│  └─────────────────┬───────────────┘       │
└────────────────────┼─────────────────────────┘
                     │
                     ▼
STAGE 3: API DEVELOPMENT
┌─────────────────────────────────────────────┐
│  FastAPI REST API                           │
│  ┌─────────────────────────────────┐        │
│  │ from fastapi import FastAPI     │        │
│  │ import joblib                   │        │
│  │                                 │        │
│  │ app = FastAPI()                 │        │
│  │ model = joblib.load('model.pkl')│        │
│  │                                 │        │
│  │ @app.post("/predict")           │        │
│  │ def predict(data: InputData):   │        │
│  │     return model.predict(data)  │        │
│  └─────────────────┬───────────────┘        │
└────────────────────┼──────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
STAGE 4: USER INTERFACE OPTIONS

┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  Streamlit   │ │   Gradio     │ │  Custom Web  │
│              │ │              │ │              │
│  Quick UI    │ │  Simple Demo │ │  Full Control│
│  Builder     │ │  Interface   │ │  HTML/CSS/JS │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       └────────────────┼────────────────┘
                        │
                        ▼
STAGE 5: CONTAINERIZATION
┌─────────────────────────────────────────────┐
│  Docker Container                           │
│  ┌─────────────────────────────────┐        │
│  │ Dockerfile                      │        │
│  │ ─────────────                   │        │
│  │ FROM python:3.10-slim          │        │
│  │ WORKDIR /app                    │        │
│  │ COPY requirements.txt .         │        │
│  │ RUN pip install -r requirements │        │
│  │ COPY . .                        │        │
│  │ CMD ["uvicorn", "app:app"]      │        │
│  └─────────────────┬───────────────┘        │
└────────────────────┼──────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│  Build & Push Image                         │
│  ┌─────────────────────────────────┐        │
│  │ docker build -t myapp:latest .  │        │
│  │ docker push myapp:latest        │        │
│  └─────────────────┬───────────────┘        │
└────────────────────┼──────────────────────────┘
                     │
                     ▼
STAGE 6: DEPLOYMENT PLATFORMS
┌───────────────────────────────────────────────────────┐
│                                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐│
│  │ Streamlit    │  │   Heroku     │  │  AWS EC2/   ││
│  │   Cloud      │  │              │  │    ECS      ││
│  │ (Free Tier)  │  │ (Free Tier)  │  │ (Scalable)  ││
│  └──────┬───────┘  └──────┬───────┘  └──────┬──────┘│
│         │                 │                  │       │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐│
│  │   Azure      │  │   Google     │  │   Railway   ││
│  │ App Service  │  │  Cloud Run   │  │             ││
│  └──────────────┘  └──────────────┘  └─────────────┘│
└───────────────────────────────────────────────────────┘
                     │
                     ▼
STAGE 7: MONITORING & MAINTENANCE
┌─────────────────────────────────────────────┐
│  Monitor Performance                        │
│  ┌─────────────────────────────────┐        │
│  │ • Application Logs              │        │
│  │ • Response Times                │        │
│  │ • Error Rates                   │        │
│  │ • Model Drift Detection         │        │
│  │ • Resource Usage (CPU, Memory)  │        │
│  └─────────────────────────────────┘        │
│                                             │
│  Tools: CloudWatch, Datadog, Prometheus    │
└─────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│  Continuous Improvement                     │
│  ┌─────────────────────────────────┐        │
│  │ • Collect user feedback         │        │
│  │ • Retrain with new data         │        │
│  │ • Update model version          │        │
│  │ • Deploy new version            │        │
│  └─────────────────────────────────┘        │
└─────────────────────────────────────────────┘
```

## Deployment Checklist

### ✅ Pre-Deployment

- [ ] Model trained and evaluated
- [ ] Model serialized (.pkl, .h5, .joblib)
- [ ] API endpoints tested locally
- [ ] Input validation implemented
- [ ] Error handling added
- [ ] Unit tests written
- [ ] Dependencies documented (requirements.txt)
- [ ] Environment variables configured

### ✅ Containerization

- [ ] Dockerfile created
- [ ] .dockerignore added
- [ ] Image builds successfully
- [ ] Container runs locally
- [ ] Image pushed to registry
- [ ] docker-compose.yml configured (if needed)

### ✅ Deployment

- [ ] Platform selected (Streamlit Cloud, Heroku, AWS, etc.)
- [ ] Secrets/API keys configured
- [ ] Environment variables set
- [ ] Deployed to staging environment
- [ ] Tested in staging
- [ ] Deployed to production
- [ ] Custom domain configured (optional)

### ✅ Post-Deployment

- [ ] Health check endpoint working
- [ ] Monitoring enabled
- [ ] Logging configured
- [ ] Alert system set up
- [ ] Backup strategy defined
- [ ] Rollback plan documented
- [ ] Performance benchmarked

## Deployment Commands Reference

### Local Development

```bash
# FastAPI
uvicorn app:app --reload --port 8000

# Streamlit
streamlit run app.py

# Gradio
python app.py
```

### Docker

```bash
# Build image
docker build -t mlm-sales-app .

# Run container
docker run -p 8000:8000 mlm-sales-app

# Using docker-compose
docker-compose up --build
```

### Streamlit Cloud

```bash
# Push to GitHub
git push origin main

# Deploy via Streamlit Cloud UI
# https://streamlit.io/cloud
```

### Heroku

```bash
heroku login
heroku create mlm-sales-app
git push heroku main
heroku open
```

### AWS EC2

```bash
# SSH into instance
ssh -i key.pem ec2-user@<instance-ip>

# Pull Docker image
docker pull your-registry/mlm-app:latest

# Run container
docker run -d -p 80:8000 your-registry/mlm-app:latest
```

## Platform Comparison

| Platform              | Cost      | Ease       | Scalability | Best For                    |
| --------------------- | --------- | ---------- | ----------- | --------------------------- |
| **Streamlit Cloud**   | Free      | ⭐⭐⭐⭐⭐ | ⭐⭐⭐      | Quick demos, prototypes     |
| **Heroku**            | Free tier | ⭐⭐⭐⭐   | ⭐⭐⭐      | Small apps, testing         |
| **Railway**           | Free tier | ⭐⭐⭐⭐   | ⭐⭐⭐⭐    | Modern apps, APIs           |
| **AWS EC2**           | Pay-as-go | ⭐⭐⭐     | ⭐⭐⭐⭐⭐  | Production, enterprise      |
| **Google Cloud Run**  | Pay-as-go | ⭐⭐⭐⭐   | ⭐⭐⭐⭐⭐  | Serverless, containers      |
| **Azure App Service** | Pay-as-go | ⭐⭐⭐     | ⭐⭐⭐⭐⭐  | Enterprise, Microsoft stack |

## Converting to Image

```bash
# Install Mermaid CLI
npm install -g @mermaid-js/mermaid-cli

# Convert to PNG
mmdc -i deployment-flow.md -o deployment-flow.png

# Convert to SVG
mmdc -i deployment-flow.md -o deployment-flow.svg
```

Or use: https://mermaid.live/
