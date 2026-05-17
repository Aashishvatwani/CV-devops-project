pipeline {
    agent any

    options {
        timestamps()
    }

    environment {
        IMAGE_NAME = "${env.IMAGE_NAME ?: 'cv-project'}"
        IMAGE_TAG = "${env.IMAGE_TAG ?: env.BUILD_NUMBER}"
        APP_PORT = "${env.APP_PORT ?: '8501'}"

        // Jenkins secret text credential ID for Cloudflare Tunnel token.
        CF_TUNNEL_TOKEN_ID = "${env.CF_TUNNEL_TOKEN_ID ?: 'cloudflare-tunnel-token'}"

        // Comma-separated env var names to inject into runtime env file.
        // Example: APP_ENV_KEYS=UNSPLASH_ACCESS_KEY
        APP_ENV_KEYS = "${env.UNSPLASH_ACCESS_KEY ? 'UNSPLASH_ACCESS_KEY' : ''}"
    }

    stages {
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
                        lines << "${k}=${env.getProperty(k)}"
                    }
                    writeFile file: '.env.runtime', text: lines.join("\n") + "\n"
                }
            }
        }

        stage('Deploy (docker compose)') {
            steps {
                script {
                    if (isUnix()) {
                        sh '''
                            if command -v docker-compose >/dev/null 2>&1; then
                                IMAGE_NAME=${IMAGE_NAME} IMAGE_TAG=${IMAGE_TAG} APP_PORT=${APP_PORT} \
                                    docker-compose -f docker-compose.yml up -d --remove-orphans
                            else
                                IMAGE_NAME=${IMAGE_NAME} IMAGE_TAG=${IMAGE_TAG} APP_PORT=${APP_PORT} \
                                    docker compose -f docker-compose.yml up -d --remove-orphans
                            fi
                        '''
                    } else {
                        bat "set IMAGE_NAME=${env.IMAGE_NAME}&& set IMAGE_TAG=${env.IMAGE_TAG}&& set APP_PORT=${env.APP_PORT}&& docker compose -f docker-compose.yml up -d --remove-orphans"
                    }
                }
            }
        }

        stage('Cloudflare tunnel') {
            steps {
                withCredentials([string(credentialsId: env.CF_TUNNEL_TOKEN_ID, variable: 'CF_TUNNEL_TOKEN')]) {
                    script {
                        if (isUnix()) {
                            sh '''
                                set -e
                                if ! command -v cloudflared >/dev/null 2>&1; then
                                    curl -L -o cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
                                    chmod +x cloudflared
                                    CLOUD_FLARED=./cloudflared
                                else
                                    CLOUD_FLARED=cloudflared
                                fi

                                pkill -f "cloudflared tunnel" >/dev/null 2>&1 || true
                                nohup "$CLOUD_FLARED" tunnel run --token "$CF_TUNNEL_TOKEN" >/var/tmp/cloudflared.log 2>&1 &
                            '''
                        } else {
                            bat "echo Cloudflare tunnel stage is supported only on Unix agents"
                        }
                    }
                }
            }
        }
    }
}
