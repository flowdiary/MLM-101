#!/usr/bin/env python3
"""
Lecture 83 - Docker and Containerization Script

Usage:
    python 04_docker_and_containerization.py --build --run
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a shell command and print output."""
    print(f"\n{description}...")
    print(f"Running: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    
    return result.returncode == 0


def build_docker_image(tag="lecture83-fastapi:latest"):
    """Build Docker image."""
    app_dir = Path(__file__).parent.parent / "apps" / "fastapi_app"
    
    if not app_dir.exists():
        print(f"Error: {app_dir} not found")
        return False
    
    cmd = f"cd {app_dir} && docker build -t {tag} ."
    return run_command(cmd, "Building Docker image")


def run_docker_container(tag="lecture83-fastapi:latest", port=8000):
    """Run Docker container."""
    models_dir = Path(__file__).parent.parent / "models"
    models_dir = models_dir.resolve()
    
    cmd = f"""docker run -d \
        --name lecture83-server \
        -p {port}:{port} \
        -v {models_dir}:/app/models \
        {tag}"""
    
    return run_command(cmd, "Starting Docker container")


def stop_container():
    """Stop and remove container."""
    run_command("docker stop lecture83-server", "Stopping container")
    run_command("docker rm lecture83-server", "Removing container")


def show_logs():
    """Show container logs."""
    run_command("docker logs -f lecture83-server", "Container logs")


def main():
    parser = argparse.ArgumentParser(description='Docker Containerization Demo')
    parser.add_argument('--build', action='store_true', help='Build Docker image')
    parser.add_argument('--run', action='store_true', help='Run Docker container')
    parser.add_argument('--stop', action='store_true', help='Stop container')
    parser.add_argument('--logs', action='store_true', help='Show logs')
    parser.add_argument('--tag', type=str, default='lecture83-fastapi:latest',
                        help='Docker image tag')
    parser.add_argument('--port', type=int, default=8000, help='Port to expose')
    
    args = parser.parse_args()
    
    if args.build:
        build_docker_image(args.tag)
    
    if args.run:
        run_docker_container(args.tag, args.port)
    
    if args.stop:
        stop_container()
    
    if args.logs:
        show_logs()
    
    if not any([args.build, args.run, args.stop, args.logs]):
        parser.print_help()
        print("\nCommon workflows:")
        print("  Build and run: python 04_docker_and_containerization.py --build --run")
        print("  Stop: python 04_docker_and_containerization.py --stop")
        print("  View logs: python 04_docker_and_containerization.py --logs")


if __name__ == "__main__":
    main()
