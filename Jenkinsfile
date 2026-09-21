pipeline {
    agent any

    environment {
        IMAGE_NAME = 'wellness-tourism-app'
        CONTAINER_NAME = 'wellness-tourism-app'
        HOST_PORT = '7860'
        CONTAINER_PORT = '7860'
    }

    stages {

        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Build Docker Image') {
            steps {
                sh '''
                    docker build -t ${IMAGE_NAME}:${BUILD_NUMBER} .
                '''
            }
        }

        stage('Stop Existing Container') {
            steps {
                sh '''
                    docker stop ${CONTAINER_NAME} || true
                    docker rm ${CONTAINER_NAME} || true
                '''
            }
        }

        stage('Deploy') {
            steps {
                sh '''
                    docker run -d \
                        --name ${CONTAINER_NAME} \
                        --restart unless-stopped \
                        -p ${HOST_PORT}:${CONTAINER_PORT} \
                        ${IMAGE_NAME}:${BUILD_NUMBER}
                '''
            }
        }

        stage('Verify Deployment') {
            steps {
                sh '''
                    sleep 10
                    curl --fail http://localhost:${HOST_PORT}/_stcore/health
                '''
            }
        }
    }

    post {
        success {
            echo 'Wellness Tourism application deployed successfully.'
        }

        failure {
            echo 'Wellness Tourism deployment failed.'
        }
    }
}