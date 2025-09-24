variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "ap-northeast-1"
}

variable "aws_profile" {
  description = "AWS profile name"
  type        = string
  default     = "default"
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "docja"
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "g6.4xlarge"
}

variable "key_name" {
  description = "EC2 key pair name"
  type        = string
}

variable "ebs_volume_size" {
  description = "EBS volume size in GB"
  type        = number
  default     = 300
}