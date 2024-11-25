![Council](council_banner.png "council")

<h1><p align="center">Council: AI Agent Platform with Control Flow and Scalable Oversight</p></h1>

![Supported Python versions](https://raw.githubusercontent.com/chain-ml/council/main/docs/source/_static/python.svg)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI version](https://badge.fury.io/py/council-ai.svg)](https://badge.fury.io/py/council-ai)
[![](https://dcbadge.vercel.app/api/server/DWNCftGQZ3?compact=true&style=flat)](https://discord.gg/DWNCftGQZ3)
[![](https://readthedocs.org/projects/council/badge/?version=stable)](https://council.dev)

# Welcome

**Council** is an open-source platform for building applications with Large Language Models (LLMs) - built in Python.

Council provides a unified interface for working with different LLM providers like OpenAI, Anthropic, Google and Ollama. The framework makes it easy to switch between providers while maintaining consistent interfaces and monitoring capabilities.

**Council** aims to provide enterprise-grade quality control and monitoring for LLM applications (contributions are welcome).

# Key Features

* 🧐 **Unified LLM Interface**: Consistent API across different LLM providers with built-in error handling and retries
* 🔄 **Provider Flexibility**: Easy switching between LLM providers like OpenAI, Anthropic, Google Gemini, and local models via Groq and Ollama
* 📊 **Usage Monitoring**: Built-in consumption tracking and monitoring capabilities
* 🛠️ **Configuration Management**: Flexible configuration system for LLM parameters like temperature, max tokens etc.
* 🔒 **Error Handling**: Robust error handling and retry mechanisms for production use

# Key Concepts

## LLM Interface

The core of Council is the LLM interface which provides a unified way to interact with different language model providers. This includes:

- Flexible configuration options
- Consistent message formatting across providers
- Built-in retry mechanisms
- Usage tracking and monitoring

## Configuration

Council provides a robust configuration system that allows you to:

- Set provider-specific parameters
- Configure retry behavior
- Control model parameters like temperature, max tokens etc.
- Manage API credentials

## Monitoring

Built-in monitoring capabilities help track:

- Token usage and costs
- Number of API calls
- Response times

# Quickstart

## Installation

Install Council in one of multiple ways:

1. (Recommended) Install with pip via Pypi: `pip install council-ai`
2. Install with pip from git ref: `pip install git+https://github.com/chain-ml/council.git@<branch_name>`
   - More documentation here: https://pip.pypa.io/en/stable/topics/vcs-support/#git
3. Install with pip from local copy: 
   - Clone this repository
   - Navigate to local project root and install via `pip install -e git+https://github.com/chain-ml/council.git@<branch_name>.`

Uninstall with: `pip uninstall council-ai`

### Current Stable Version
<a href="https://pypi.org/project/council-ai/#history"><img alt="GitHub release (latest SemVer)" src="https://img.shields.io/github/v/release/chain-ml/council"></a>


## Setup

Set up your required API keys in a `.env` file (e.g. OpenAI). Refer to `.env.example` as an example.

## Linter

Use `make lint` to verify your code.

## Black

Use `black .` to automatically reformat files.

# Documentation

A detailed documentation of Council can be found at <a href="https://council.dev">council.dev</a>.

# Support

Please submit a GitHub issue should you need any help or reach out to the team via <a href="https://discord.gg/DWNCftGQZ3">Discord</a>.

# Contributors

Council is a project under active development. We welcome all contributions, pull requests, feature requests or reported issues.
