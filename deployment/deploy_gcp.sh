#!/bin/bash
# GCP Deployment Script for Distributed Traffic Control System
# Provisions GCP resources and deploys containers

set -e

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-traffic-control-system}"
REGION="${GCP_REGION:-us-central1}"
ZONE="${GCP_ZONE:-us-central1-a}"
VM_NAME="traffic-controller-vm"
VM_TYPE="e2-standard-4"
DISK_SIZE="50GB"
IMAGE_FAMILY="ubuntu-2204-lts"
IMAGE_PROJECT="ubuntu-os-cloud"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI not found. Please install Google Cloud SDK."
        exit 1
    fi
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker not found. Please install Docker."
        exit 1
    fi
    
    log_info "Prerequisites OK"
}

# Authenticate with GCP
authenticate() {
    log_info "Authenticating with GCP..."
    
    # Check if already authenticated
    if gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null | grep -q "@"; then
        log_info "Already authenticated"
    else
        gcloud auth login
    fi
    
    gcloud config set project $PROJECT_ID
    gcloud config set compute/zone $ZONE
    gcloud config set compute/region $REGION
}

# Create VM instance
create_vm() {
    log_info "Creating VM instance: $VM_NAME..."
    
    # Check if VM already exists
    if gcloud compute instances describe $VM_NAME --zone=$ZONE &> /dev/null; then
        log_warn "VM $VM_NAME already exists. Skipping creation."
        return
    fi
    
    gcloud compute instances create $VM_NAME \
        --zone=$ZONE \
        --machine-type=$VM_TYPE \
        --boot-disk-size=$DISK_SIZE \
        --image-family=$IMAGE_FAMILY \
        --image-project=$IMAGE_PROJECT \
        --tags=http-server,https-server,mqtt-server \
        --metadata=startup-script='#!/bin/bash
            apt-get update
            apt-get install -y docker.io docker-compose
            systemctl start docker
            systemctl enable docker
            usermod -aG docker $USER
        '
    
    log_info "VM created successfully"
}

# Create firewall rules
create_firewall_rules() {
    log_info "Creating firewall rules..."
    
    # MQTT port
    if ! gcloud compute firewall-rules describe allow-mqtt &> /dev/null; then
        gcloud compute firewall-rules create allow-mqtt \
            --allow=tcp:1883,tcp:8883 \
            --target-tags=mqtt-server \
            --description="Allow MQTT traffic"
    fi
    
    # PostgreSQL port (internal only recommended)
    if ! gcloud compute firewall-rules describe allow-postgres &> /dev/null; then
        gcloud compute firewall-rules create allow-postgres \
            --allow=tcp:5432 \
            --source-ranges="10.0.0.0/8" \
            --description="Allow PostgreSQL internal traffic"
    fi
    
    log_info "Firewall rules configured"
}

# Create Cloud SQL instance (optional - can use containerized PostgreSQL)
create_cloud_sql() {
    log_info "Setting up Cloud SQL (optional)..."
    
    SQL_INSTANCE="${PROJECT_ID}-sql"
    
    # Check if instance exists
    if gcloud sql instances describe $SQL_INSTANCE &> /dev/null 2>&1; then
        log_warn "Cloud SQL instance already exists"
        return
    fi
    
    read -p "Create Cloud SQL instance? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Skipping Cloud SQL - using containerized PostgreSQL"
        return
    fi
    
    gcloud sql instances create $SQL_INSTANCE \
        --database-version=POSTGRES_15 \
        --tier=db-f1-micro \
        --region=$REGION \
        --root-password=traffic_secure_pwd
    
    # Create database
    gcloud sql databases create traffic_control --instance=$SQL_INSTANCE
    
    log_info "Cloud SQL created"
}

# Deploy containers to VM
deploy_containers() {
    log_info "Deploying containers to VM..."
    
    # Get VM external IP
    VM_IP=$(gcloud compute instances describe $VM_NAME --zone=$ZONE \
        --format='get(networkInterfaces[0].accessConfigs[0].natIP)')
    
    log_info "VM IP: $VM_IP"
    
    # Copy docker-compose and source files
    log_info "Copying files to VM..."
    
    gcloud compute scp --zone=$ZONE --recurse \
        ../cloud $VM_NAME:~/traffic-control/cloud
    
    gcloud compute scp --zone=$ZONE \
        ../sioux.net.xml $VM_NAME:~/traffic-control/
    
    # SSH and run docker-compose
    log_info "Starting containers..."
    
    gcloud compute ssh $VM_NAME --zone=$ZONE --command="
        cd ~/traffic-control/cloud
        docker-compose build
        docker-compose up -d
    "
    
    log_info "Containers deployed"
    log_info "MQTT Broker: $VM_IP:1883"
    log_info "PostgreSQL: $VM_IP:5432"
}

# Print connection info
print_info() {
    VM_IP=$(gcloud compute instances describe $VM_NAME --zone=$ZONE \
        --format='get(networkInterfaces[0].accessConfigs[0].natIP)' 2>/dev/null || echo "N/A")
    
    echo ""
    echo "=============================================="
    echo "  DEPLOYMENT COMPLETE"
    echo "=============================================="
    echo ""
    echo "VM Name:     $VM_NAME"
    echo "VM IP:       $VM_IP"
    echo "Project:     $PROJECT_ID"
    echo "Region:      $REGION"
    echo ""
    echo "Services:"
    echo "  MQTT Broker:    mqtt://$VM_IP:1883"
    echo "  PostgreSQL:     postgresql://traffic_admin:***@$VM_IP:5432/traffic_control"
    echo ""
    echo "To SSH into VM:"
    echo "  gcloud compute ssh $VM_NAME --zone=$ZONE"
    echo ""
    echo "To view logs:"
    echo "  gcloud compute ssh $VM_NAME --zone=$ZONE --command='cd ~/traffic-control/cloud && docker-compose logs -f'"
    echo ""
    echo "=============================================="
}

# Cleanup function
cleanup() {
    log_warn "Cleaning up resources..."
    
    read -p "Are you sure you want to delete all resources? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Cleanup cancelled"
        return
    fi
    
    # Delete VM
    gcloud compute instances delete $VM_NAME --zone=$ZONE --quiet || true
    
    # Delete firewall rules
    gcloud compute firewall-rules delete allow-mqtt --quiet || true
    gcloud compute firewall-rules delete allow-postgres --quiet || true
    
    log_info "Cleanup complete"
}

# Main execution
main() {
    case "${1:-deploy}" in
        deploy)
            check_prerequisites
            authenticate
            create_vm
            create_firewall_rules
            # create_cloud_sql  # Uncomment to use Cloud SQL instead of container
            
            # Wait for VM startup script
            log_info "Waiting for VM to be ready (60 seconds)..."
            sleep 60
            
            deploy_containers
            print_info
            ;;
        cleanup)
            cleanup
            ;;
        info)
            print_info
            ;;
        *)
            echo "Usage: $0 {deploy|cleanup|info}"
            exit 1
            ;;
    esac
}

main "$@"
