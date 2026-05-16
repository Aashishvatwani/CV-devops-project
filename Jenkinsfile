pipeline {
    agent any

    options {
        timestamps()
    }

    environment {
        IMAGE_NAME = "${env.IMAGE_NAME ?: 'cv-project'}"
        IMAGE_TAG = "${env.IMAGE_TAG ?: env.BUILD_NUMBER}"
        APP_PORT = "${env.APP_PORT ?: '8501'}"

        // Comma-separated env var names to inject into runtime env file.
        // Example: APP_ENV_KEYS=UNSPLASH_ACCESS_KEY
        APP_ENV_KEYS = "${env.UNSPLASH_ACCESS_KEY ? 'UNSPLASH_ACCESS_KEY' : ''}"
    }

    stages {
        stage('Validate env') {
            steps {
                script {
                    def keys = (env.APP_ENV_KEYS ?: '')
                        .split(',')
                        .collect { it.trim() }
                        .findAll { it }
                    def missing = keys.findAll { !env[it] }
                    if (missing) {
                        error("Missing Jenkins global env vars: ${missing.join(', ')}")
                    }
                }
            }
        }

        stage('Python venv') {
            steps {
                script {
                    if (isUnix()) {
                        sh '''
                            python3 -m venv .venv
                            . .venv/bin/activate
                            python -m pip install --upgrade pip
                            pip install -r requirements.txt
                        '''
                    } else {
                        bat '''
                            python -m venv .venv
                            call .venv\\Scripts\\activate
                            python -m pip install --upgrade pip
                            pip install -r requirements.txt
                        '''
                    }
                }
            }
        }

        stage('Build image') {
            steps {
                script {
                    if (isUnix()) {
                        sh "docker build -t ${env.IMAGE_NAME}:${env.IMAGE_TAG} ."
                    } else {
                        bat "docker build -t ${env.IMAGE_NAME}:${env.IMAGE_TAG} ."
                    }
                }
            }
        }
        stage('Write runtime env') {
            steps {
                script {
                    def keys = (env.APP_ENV_KEYS ?: '')
                        .split(',')
                        .collect { it.trim() }
                        .findAll { it }
                    def lines = []
                    for (k in keys) {
                        lines << "${k}=${env[k]}"
                    }
                    writeFile file: '.env.runtime', text: lines.join("\n") + "\n"
                }
            }
        }

        stage('Deploy (docker compose)') {
            steps {
                script {
                    if (isUnix()) {
                        sh "IMAGE_NAME=${env.IMAGE_NAME} IMAGE_TAG=${env.IMAGE_TAG} APP_PORT=${env.APP_PORT} docker compose -f docker-compose.yml up -d --remove-orphans"
                    } else {
                        bat "set IMAGE_NAME=${env.IMAGE_NAME}&& set IMAGE_TAG=${env.IMAGE_TAG}&& set APP_PORT=${env.APP_PORT}&& docker compose -f docker-compose.yml up -d --remove-orphans"
                    }
                }
            }
        }
    }
}
