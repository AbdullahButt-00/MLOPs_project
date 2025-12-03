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
        NAMESPACE = 'churn-prediction'
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
                    if pgrep -f "mlflow server" > /dev/null; then
                        echo "✓ MLflow server already running"
                    else
                        echo "Starting MLflow server..."
                        nohup mlflow server --host 0.0.0.0 --port 5000 > mlflow.log 2>&1 &
                        sleep 5
                    fi
                    
                    # Test connection
                    for i in {1..5}; do
                        if curl -f http://localhost:5000/health 2>/dev/null; then
                            echo "✓ MLflow is accessible"
                            break
                        fi
                        echo "Waiting for MLflow... ($i/5)"
                        sleep 2
                    done
                '''
            }
        }
        
        stage('Build Docker Images in Minikube') {
            steps {
                echo '🐳 Building Docker images inside Minikube...'
                sh '''
                    # Use Minikube's Docker daemon
                    eval $(minikube docker-env)
                    
                    echo "Building preprocessing image..."
                    docker build -t churn-preprocess:latest -f Dockerfile.preprocess .
                    
                    echo "Building training image..."
                    docker build -t churn-training:latest -f Dockerfile.training .
                    
                    echo "✓ Preprocessing and training images built"
                '''
            }
        }
        
        stage('Data Preprocessing') {
            steps {
                echo '🔄 Running data preprocessing...'
                sh '''
                    # Use Minikube's Docker to run preprocessing
                    eval $(minikube docker-env)
                    
                    docker run --rm \
                        -v $(pwd):/app \
                        -w /app \
                        churn-preprocess:latest \
                        python preprocess.py \
                            --dataset ${DATASET} \
                            --output-folder preprocessed_data \
                            --clients 3
                    
                    # Verify output
                    if [ ! -f "preprocessed_data/preprocessor.pkl" ]; then
                        echo "❌ Preprocessing failed"
                        exit 1
                    fi
                    
                    echo "✓ Preprocessing completed"
                    ls -lh preprocessed_data/
                '''
            }
        }
        
        stage('Model Training') {
            steps {
                echo '🤖 Training federated model...'
                sh '''
                    # Use Minikube's Docker to run training
                    eval $(minikube docker-env)
                    
                    docker run --rm \
                        -v $(pwd):/app \
                        -w /app \
                        churn-training:latest \
                        python training_MLFlow.py
                    
                    # Verify model was created
                    if [ ! -f "federated_data/federated_churn_model.h5" ]; then
                        echo "❌ Training failed"
                        exit 1
                    fi
                    
                    echo "✓ Training completed"
                    ls -lh federated_data/
                '''
            }
        }
        
        stage('Build Serving Image') {
            steps {
                echo '🐳 Building serving image (after model files created)...'
                sh '''
                    # Use Minikube's Docker daemon
                    eval $(minikube docker-env)
                    
                    echo "Building serving image..."
                    docker build -t churn-serving:latest -f Dockerfile.serving .
                    
                    echo "Verifying all images in Minikube:"
                    docker images | grep churn
                    
                    echo "✓ All images built in Minikube"
                '''
            }
        }
        
        stage('Extract Model Metrics') {
            steps {
                echo '📊 Extracting model metrics...'
                sh '''
                    if [ -f "federated_data/round_evaluation/per_round_metrics.csv" ]; then
                        ACCURACY=$(tail -1 federated_data/round_evaluation/per_round_metrics.csv | cut -d',' -f3)
                        echo "Model Accuracy: ${ACCURACY}"
                        
                        if [ "${SKIP_TESTS}" = "false" ]; then
                            PASS=$(echo "${ACCURACY} >= 0.75" | bc -l)
                            if [ "$PASS" -eq 0 ]; then
                                echo "❌ Accuracy ${ACCURACY} below 0.75"
                                exit 1
                            fi
                        fi
                        
                        echo "✓ Model accuracy acceptable"
                    else
                        echo "⚠️  Metrics file not found"
                    fi
                '''
            }
        }
        
        stage('Copy Model Data to PVC') {
            when {
                expression { params.FORCE_DEPLOY || currentBuild.result == null }
            }
            steps {
                echo '📦 Copying model data to Kubernetes PVC...'
                sh '''
                    # Delete old copier pod if exists
                    kubectl delete pod data-copier -n ${NAMESPACE} 2>/dev/null || true
                    sleep 2
                    
                    # Create copier pod
                    kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: data-copier
  namespace: ${NAMESPACE}
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
                    
                    # Wait for pod
                    kubectl wait --for=condition=Ready pod/data-copier -n ${NAMESPACE} --timeout=60s
                    
                    # Copy data
                    echo "Copying preprocessed data..."
                    kubectl cp preprocessed_data ${NAMESPACE}/data-copier:/data/
                    
                    echo "Copying federated data..."
                    kubectl cp federated_data ${NAMESPACE}/data-copier:/data/
                    
                    # Verify
                    echo "Verifying files..."
                    kubectl exec -n ${NAMESPACE} data-copier -- sh -c "ls -lh /data/preprocessed_data && ls -lh /data/federated_data"
                    
                    echo "✓ Data copied successfully"
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
                    # Apply all configurations
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
                    kubectl apply -f k8s/api-deployment.yaml
                    
                    echo "✓ Deployments applied"
                '''
            }
        }
        
        stage('Wait for Deployments') {
            when {
                expression { params.FORCE_DEPLOY || currentBuild.result == null }
            }
            steps {
                echo '⏳ Waiting for deployments to be ready...'
                sh '''
                    echo "Waiting for API deployment..."
                    kubectl wait --for=condition=available --timeout=180s deployment/churn-api -n ${NAMESPACE} || {
                        echo "⚠️  API deployment timeout"
                        kubectl get pods -n ${NAMESPACE} -l app=churn-api
                        kubectl describe pods -n ${NAMESPACE} -l app=churn-api | tail -50
                        exit 1
                    }
                    
                    echo "Waiting for monitoring stack..."
                    kubectl wait --for=condition=available --timeout=120s deployment/prometheus -n ${NAMESPACE} || true
                    kubectl wait --for=condition=available --timeout=120s deployment/grafana -n ${NAMESPACE} || true
                    
                    echo "✓ Deployments ready"
                '''
            }
        }
        
        stage('Health Check & Generate Traffic') {
            steps {
                echo '🏥 Testing deployment...'
                sh '''
                    # Get service URL
                    API_URL=$(minikube service churn-api-service -n ${NAMESPACE} --url)
                    echo "API URL: $API_URL"
                    
                    # Wait for API to respond
                    for i in {1..30}; do
                        if curl -f -s ${API_URL}/health >/dev/null 2>&1; then
                            echo "✓ API is healthy"
                            break
                        fi
                        echo "Waiting for API... ($i/30)"
                        sleep 5
                    done
                    
                    # Test prediction
                    echo "Testing prediction endpoint..."
                    curl -X POST ${API_URL}/predict \
                        -H "Content-Type: application/json" \
                        -d '{"Tenure": 12.0, "PreferredLoginDevice": "Mobile Phone", "CityTier": 1, "WarehouseToHome": 15.0, "PreferredPaymentMode": "Credit Card", "Gender": "Male", "HourSpendOnApp": 3.0, "NumberOfDeviceRegistered": 3, "PreferedOrderCat": "Laptop & Accessory", "SatisfactionScore": 5, "MaritalStatus": "Single", "NumberOfAddress": 2, "Complain": 0, "OrderAmountHikeFromlastYear": 15.0, "CouponUsed": 1.0, "OrderCount": 5.0, "DaySinceLastOrder": 3.0, "CashbackAmount": 150.0}'
                    
                    echo ""
                    echo "Generating test traffic (20 requests)..."
                    for i in {1..20}; do
                        curl -X POST ${API_URL}/predict \
                            -H "Content-Type: application/json" \
                            -d '{"Tenure": 12.0, "PreferredLoginDevice": "Mobile Phone", "CityTier": 1, "WarehouseToHome": 15.0, "PreferredPaymentMode": "Credit Card", "Gender": "Male", "HourSpendOnApp": 3.0, "NumberOfDeviceRegistered": 3, "PreferedOrderCat": "Laptop & Accessory", "SatisfactionScore": 5, "MaritalStatus": "Single", "NumberOfAddress": 2, "Complain": 0, "OrderAmountHikeFromlastYear": 15.0, "CouponUsed": 1.0, "OrderCount": 5.0, "DaySinceLastOrder": 3.0, "CashbackAmount": 150.0}' \
                            -s >/dev/null && echo "  ✓ $i" || echo "  ✗ $i"
                        sleep 0.3
                    done
                    
                    echo "✓ Traffic generated"
                '''
            }
        }
        
        stage('Cleanup Temporary Resources') {
            steps {
                echo '🧹 Cleaning up temporary resources...'
                sh '''
                    # Delete data copier pod
                    kubectl delete pod data-copier -n ${NAMESPACE} 2>/dev/null || true
                    
                    # Reset Docker environment
                    eval $(minikube docker-env -u)
                    
                    echo "✓ Cleanup complete"
                '''
            }
        }
    }
    
    post {
        always {
            echo '📋 Archiving artifacts...'
            archiveArtifacts artifacts: 'federated_data/**/*.png,federated_data/**/*.csv,federated_data/**/*.h5,preprocessed_data/**/*.pkl', allowEmptyArchive: true
        }
        
        success {
            echo '''
            ============================================================
            ✅ PIPELINE COMPLETED SUCCESSFULLY
            ============================================================
            
            Model trained and deployed to Kubernetes!
            
            Access services:
            - API: minikube service churn-api-service -n churn-prediction
            - Grafana: minikube service grafana-service -n churn-prediction  
            - Prometheus: minikube service prometheus-service -n churn-prediction
            
            Grafana credentials: admin / root
            
            ============================================================
            '''
        }
        
        failure {
            echo '''
            ============================================================
            ❌ PIPELINE FAILED
            ============================================================
            
            Check logs above for details.
            
            Debugging commands:
            - kubectl get pods -n churn-prediction
            - kubectl logs -n churn-prediction -l app=churn-api
            - kubectl describe deployment churn-api -n churn-prediction
            
            ============================================================
            '''
        }
    }
}