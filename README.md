# Libras Vision AI

AI-powered Brazilian Sign Language (Libras) recognition system using Computer Vision and Machine Learning.

## Overview

This project aims to improve accessibility for the deaf community by translating Libras gestures into text in real time using Artificial Intelligence.

The system uses:
- Computer Vision
- Hand Landmark Detection
- Machine Learning
- Real-time gesture prediction

Future applications include:
- Customer service
- Healthcare assistance
- Ride-sharing communication
- Public service accessibility

---

## Technologies

### AI & Computer Vision
- Python
- OpenCV
- MediaPipe
- Scikit-Learn
- NumPy
- Pandas

### Backend (future architecture)
- Java
- Spring Boot
- JWT Authentication
- WebSockets
- PostgreSQL

### Infrastructure (future)
- Docker
- Kubernetes

---

## Current Features

- Real-time hand detection
- Hand landmark extraction
- Dataset collection pipeline
- Machine Learning training pipeline
- Real-time gesture prediction
- Model persistence with Joblib

---

## Project Architecture

Frontend (future)
↓
Java Spring Boot API Gateway
↓
Python AI Microservice
↓
Machine Learning Prediction

---

## Machine Learning Pipeline

Camera Input
↓
MediaPipe Hand Tracking
↓
Landmark Extraction (21 points)
↓
Dataset Generation
↓
Model Training
↓
Real-time Prediction

---

## Dataset Structure

Each gesture sample contains:
- 21 hand landmarks
- x, y, z coordinates
- 63 features per sample

---

## Goals

- Real-time Libras translation
- Multi-user gesture recognition
- AI-powered accessibility solution
- Production-ready microservice architecture

---

## Future Improvements

- Deep Learning models (LSTM / CNN)
- Sentence recognition
- Voice synthesis
- Mobile application
- Cloud deployment
- Kubernetes orchestration

---

## Author

Polyana SS

Computer Science / AI Enthusiast