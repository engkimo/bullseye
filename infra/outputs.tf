output "s3_bucket_name" {
  description = "Name of the S3 bucket for checkpoints"
  value       = aws_s3_bucket.checkpoints.id
}

output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.main.id
}

output "subnet_id" {
  description = "ID of the public subnet"
  value       = aws_subnet.public.id
}

output "security_group_id" {
  description = "ID of the security group"
  value       = aws_security_group.main.id
}

output "launch_template_id" {
  description = "ID of the launch template"
  value       = aws_launch_template.gpu.id
}

output "launch_template_latest_version" {
  description = "Latest version of the launch template"
  value       = aws_launch_template.gpu.latest_version
}

output "iam_role_arn" {
  description = "ARN of the IAM role"
  value       = aws_iam_role.ec2.arn
}