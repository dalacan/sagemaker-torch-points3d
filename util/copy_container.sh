
# Pass the docker file
docker_file=$1

# The argument to this script is the image name. This will be used as the image on the local
# machine and combined with the account and region to form the repository name for ECR.
image=$2

registry_alias=$3

# Get the account number associated with the current IAM credentials
account=$(aws sts get-caller-identity --query Account --output text)

# Get the region defined in the current configuration (default to us-west-2 if none defined)
region=$(aws configure get region)
region=${region:-us-east-1}

fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:latest"

# If the repository doesn't exist in ECR, create it.
aws ecr describe-repositories --repository-names "${image}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${image}" > /dev/null
fi

public_container="public.ecr.aws/${registry_alias}"
public_image="public.ecr.aws/${registry_alias}/${image}:latest"
aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin ${public_container}

# Pull public image
docker pull ${public_image}

# Get the login command from ECR and execute it directly
$(aws ecr get-login --region ${region} --no-include-email)

# Get the login command from ECR in order to pull down the SageMaker PyTorch image
$(aws ecr get-login --registry-ids 763104351884 --region ${region} --no-include-email)


# Tag and push to ECR
docker tag ${public_image} ${fullname}
docker push ${fullname}
