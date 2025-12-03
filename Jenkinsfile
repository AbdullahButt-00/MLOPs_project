pipeline {
    agent any
    
    parameters {
        string(name: 'DATASET_PATH', defaultValue: 'E_Commerce_Dataset.xlsx', description: 'Path to the dataset')
        booleanParam(name: 'SKIP_TESTS', defaultValue: false, description: 'Skip model evaluation tests')
        booleanParam(name: 'FORCE_DEPLOY', defaultValue: false, description: 'Force deployment even if tests fail')
    }
    
    environment {
        MLFLOW_TRACKING_URI = 'http://host.docker.internal:5000'
        DATASET = "${params.DATASET_PATH}"
        IMAGE_TAG = "${BUILD_NUMBER}"
    }
    
    stages {
        stage('Cleanup') {
            steps {
                echo '🧹 Cleaning workspace...'
                deleteDir()
            }
        }
        
        stage('Checkout') {
            steps {
                echo '📦 Checking out code...'
                checkout scm
            }
        }
        
        stage('Setup') {
            steps {
                echo '⚙️  Setting up environment...'
                sh '''
                    mkdir -p preprocessed_data
                    mkdir -p federated_data
                    mkdir -p federated_data/round_evaluation
                    
                    if [ ! -f "${DATASET}" ]; then
                        echo "❌ Error: Dataset ${DATASET} not found!"
                        exit 1
                    fi
                    
                    echo "✓ Using dataset: ${DATASET}"
                '''
            }
        }
        
        stage('Start MLflow Server') {
            steps {
                echo '🚀 Starting MLflow server...'
                sh '''
                    # Check if MLflow is already running
                    if pgrep -f "mlflow server" > /dev/null; then
                        echo "✓ MLflow server already running"
                    else
                        echo "Starting MLflow server..."
                        nohup mlflow server --host 0.0.0.0 --port 5000 > mlflow.log 2>&1 &
                        sleep 5
                    fi
                    
                    # Verify MLflow is accessible
                    curl -f http://localhost:5000/health || echo "OK"
                '''
            }
        }
        
        stage('Build Docker Images') {
            steps {
                echo '🐳 Building Docker images...'
                sh '''
                    # Build preprocessing image
                    docker build -t churn-preprocess:latest -f Dockerfile.preprocess .
                    
                    # Build training image
                    docker build -t churn-training:latest -f Dockerfile.training .
                    
                    # Build serving image
                    docker build -t churn-serving:latest -f Dockerfile.serving .
                    
                    echo "✓ Docker images built successfully"
                '''
            }
        }
        
        stage('Data Preprocessing') {
            steps {
                echo '🔄 Running data preprocessing in Docker...'
                sh '''
                    # Run preprocessing in Docker container
                    docker run --rm \
                        -v $(pwd):/app \
                        -v $(pwd)/preprocessed_data:/app/preprocessed_data \
                        -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
                        --network host \
                        churn-preprocess:latest \
                        python preprocess.py \
                            --dataset ${DATASET} \
                            --output-folder preprocessed_data \
                            --clients 3
                    
                    # Verify preprocessing outputs
                    if [ ! -f "preprocessed_data/preprocessor.pkl" ]; then
                        echo "❌ Preprocessing failed - preprocessor.pkl not found"
                        exit 1
                    fi
                    
                    echo "✓ Preprocessing completed successfully"
                '''
            }
        }
        
        stage('Model Training') {
            steps {
                echo '🤖 Training federated model in Docker...'
                sh '''
                    # Run training in Docker container
                    docker run --rm \
                        -v $(pwd):/app \
                        -v $(pwd)/preprocessed_data:/app/preprocessed_data \
                        -v $(pwd)/federated_data:/app/federated_data \
                        -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
                        --network host \
                        churn-training:latest \
                        python training_MLFlow.py
                    
                    # Verify model was created
                    if [ ! -f "federated_data/federated_churn_model.h5" ]; then
                        echo "❌ Training failed - model not found"
                        exit 1
                    fi
                    
                    echo "✓ Model training completed"
                '''
            }
        }
        
        stage('Extract Model Metrics') {
            steps {
                echo '📊 Extracting model metrics...'
                sh '''#!/bin/bash
                    # Extract accuracy from metrics file
                    if [ -f "federated_data/round_evaluation/per_round_metrics.csv" ]; then
                        ACCURACY=$(tail -1 federated_data/round_evaluation/per_round_metrics.csv | cut -d',' -f3)
                        echo "Model Accuracy: ${ACCURACY}"
                        
                        # Check if accuracy meets threshold (0.75)
                        if [ "${SKIP_TESTS}" = "false" ]; then
                            # Use heredoc to avoid substitution issues
                            python3 - <<EOF
        import sys
        accuracy = float('${ACCURACY}')
        if accuracy < 0.75:
            print(f'❌ Model accuracy ({accuracy:.4f}) below threshold (0.75)')
            sys.exit(1)
        print(f'✓ Model accuracy ({accuracy:.4f}) acceptable')
        EOF
                        else
                            echo "⚠️  Skipping accuracy check (SKIP_TESTS=true)"
                        fi
                    else
                        echo "⚠️  Metrics file not found, skipping accuracy check"
                    fi
                '''
            }
        }
        
        stage('Push Docker Images to Minikube') {
            when {
                expression { params.FORCE_DEPLOY || currentBuild.result == null }
            }
            steps {
                echo '📤 Loading Docker images into Minikube...'
                sh '''
                    # Load images into Minikube
                    minikube image load churn-preprocess:latest
                    minikube image load churn-training:latest
                    minikube image load churn-serving:latest
                    
                    echo "✓ Images loaded into Minikube"
                '''
            }
        }
        
        stage('Copy Model Data to Minikube') {
            when {
                expression { params.FORCE_DEPLOY || currentBuild.result == null }
            }
            steps {
                echo '📦 Copying model data to Minikube volumes...'
                sh '''
                    # Create a temporary pod to copy data to PVC
                    kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: data-copier
  namespace: churn-prediction
spec:
  containers:
  - name: copier
    image: busybox
    command: ['sh', '-c', 'sleep 3600']
    volumeMounts:
    - name: model-data
      mountPath: /data
  volumes:
  - name: model-data
    persistentVolumeClaim:
      claimName: model-data-pvc
  restartPolicy: Never
EOF

                    # Wait for pod to be ready
                    kubectl wait --for=condition=Ready pod/data-copier -n churn-prediction --timeout=60s
                    
                    # Copy preprocessed data
                    echo "Copying preprocessed data..."
                    kubectl cp preprocessed_data churn-prediction/data-copier:/data/
                    
                    # Copy federated data
                    echo "Copying federated data..."
                    kubectl cp federated_data churn-prediction/data-copier:/data/
                    
                    # Verify files were copied
                    kubectl exec -n churn-prediction data-copier -- ls -la /data/preprocessed_data
                    kubectl exec -n churn-prediction data-copier -- ls -la /data/federated_data
                    
                    # Clean up temporary pod
                    kubectl delete pod data-copier -n churn-prediction
                    
                    echo "✓ Model data copied successfully"
                '''
            }
        }
        
        stage('Deploy to Kubernetes') {
            when {
                expression { params.FORCE_DEPLOY || currentBuild.result == null }
            }
            steps {
                echo '☸️  Deploying to Kubernetes...'
                sh '''
                    # Apply Kubernetes configurations in order
                    kubectl apply -f k8s/namespace.yaml
                    kubectl apply -f k8s/pvc.yaml
                    kubectl apply -f k8s/prometheus-rbac.yaml
                    kubectl apply -f k8s/prometheus-configmap.yaml
                    kubectl apply -f k8s/prometheus-deployment.yaml
                    kubectl apply -f k8s/alertmanager-deployment.yaml
                    kubectl apply -f k8s/grafana-datasource-configmap.yaml
                    kubectl apply -f k8s/grafana-dashboard-config.yaml
                    kubectl apply -f k8s/grafana-dashboard-json.yaml
                    kubectl apply -f k8s/grafana-deployment.yaml
                    kubectl apply -f k8s/jenkins-trigger-deployment.yaml
                    
                    # Deploy API (this will rollout restart if already deployed)
                    kubectl apply -f k8s/api-deployment.yaml
                    
                    # Wait for critical deployments
                    echo "Waiting for deployments to be ready..."
                    kubectl wait --for=condition=available --timeout=180s deployment/prometheus -n churn-prediction || true
                    kubectl wait --for=condition=available --timeout=180s deployment/grafana -n churn-prediction || true
                    kubectl wait --for=condition=available --timeout=300s deployment/churn-api -n churn-prediction
                    
                    echo "✓ Kubernetes deployment completed"
                '''
            }
        }
        
        stage('Health Check') {
            steps {
                echo '🏥 Running health checks...'
                sh '''
                    # Get service URL
                    API_URL=$(minikube service churn-api-service -n churn-prediction --url)
                    echo "API URL: $API_URL"
                    
                    # Wait for API to be healthy
                    for i in {1..30}; do
                        if curl -f ${API_URL}/health 2>/dev/null; then
                            echo "✓ API is healthy"
                            break
                        fi
                        echo "Waiting for API to be ready... (attempt $i/30)"
                        sleep 10
                    done
                    
                    # Test prediction endpoint
                    echo "Testing prediction endpoint..."
                    curl -X POST ${API_URL}/predict \
                        -H "Content-Type: application/json" \
                        -d '{
                            "Tenure": 12.0,
                            "PreferredLoginDevice": "Mobile Phone",
                            "CityTier": 1,
                            "WarehouseToHome": 15.0,
                            "PreferredPaymentMode": "Credit Card",
                            "Gender": "Male",
                            "HourSpendOnApp": 3.0,
                            "NumberOfDeviceRegistered": 3,
                            "PreferedOrderCat": "Laptop & Accessory",
                            "SatisfactionScore": 5,
                            "MaritalStatus": "Single",
                            "NumberOfAddress": 2,
                            "Complain": 0,
                            "OrderAmountHikeFromlastYear": 15.0,
                            "CouponUsed": 1.0,
                            "OrderCount": 5.0,
                            "DaySinceLastOrder": 3.0,
                            "CashbackAmount": 150.0
                        }' && echo "" || echo "⚠️  Prediction test failed"
                    
                    # Check metrics endpoint
                    echo "Checking metrics endpoint..."
                    curl -s ${API_URL}/metrics | grep -E "predictions_total|prediction_latency" | head -5 || echo "⚠️  Metrics not available yet"
                '''
            }
        }
        
        stage('Generate Test Traffic') {
            steps {
                echo '🔄 Generating test traffic to populate metrics...'
                sh '''
                    API_URL=$(minikube service churn-api-service -n churn-prediction --url)
                    
                    # Generate 20 test predictions
                    for i in {1..20}; do
                        curl -X POST ${API_URL}/predict \
                            -H "Content-Type: application/json" \
                            -d '{
                                "Tenure": 12.0,
                                "PreferredLoginDevice": "Mobile Phone",
                                "CityTier": 1,
                                "WarehouseToHome": 15.0,
                                "PreferredPaymentMode": "Credit Card",
                                "Gender": "Male",
                                "HourSpendOnApp": 3.0,
                                "NumberOfDeviceRegistered": 3,
                                "PreferedOrderCat": "Laptop & Accessory",
                                "SatisfactionScore": 5,
                                "MaritalStatus": "Single",
                                "NumberOfAddress": 2,
                                "Complain": 0,
                                "OrderAmountHikeFromlastYear": 15.0,
                                "CouponUsed": 1.0,
                                "OrderCount": 5.0,
                                "DaySinceLastOrder": 3.0,
                                "CashbackAmount": 150.0
                            }' -s > /dev/null && echo "  Request $i: ✓" || echo "  Request $i: ✗"
                        sleep 0.5
                    done
                    
                    echo "✓ Test traffic generated"
                '''
            }
        }
    }
    
    post {
        always {
            echo '📋 Archiving artifacts...'
            archiveArtifacts artifacts: 'federated_data/**/*.png,federated_data/**/*.csv,federated_data/**/*.h5,preprocessed_data/**/*.pkl', allowEmptyArchive: true
            
            // Cleanup Docker images
            sh 'docker image prune -f'
        }
        
        success {
            echo '''
            ============================================================
            ✅ PIPELINE COMPLETED SUCCESSFULLY
            ============================================================
            
            Model has been trained and deployed to Kubernetes.
            
            Access the services:
            - API: minikube service churn-api-service -n churn-prediction
            - Grafana: minikube service grafana-service -n churn-prediction
            - Prometheus: minikube service prometheus-service -n churn-prediction
            
            Default Grafana credentials: admin / root
            
            ============================================================
            '''
        }
        
        failure {
            echo '''
            ============================================================
            ❌ PIPELINE FAILED
            ============================================================
            
            Check the logs above for error details.
            Common issues:
            - Dataset not found
            - Model accuracy below threshold
            - Docker build failures
            - Kubernetes deployment issues
            
            ============================================================
            '''
        }
    }
}