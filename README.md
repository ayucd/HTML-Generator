# HTML Generator

This is a fine-tuned Llama-2-7B model that generates HTML code for a given natural language prompt. The fine-tuned was completed using the Supervised Fine Tuning Trainer (SFTTrainer) from the Transformer Reinforcement Learning (trl) library, which is a full stack library that provides a set of tools to train language models with Reinforcement Learning, from the Supervised Fine-tuning step (SFT), Reward Modeling step (RM) to the Proximal Policy Optimization (PPO) step. The library is integrated with Hugging Face transformers.

## Index
1. [The Dataset](#the-dataset)
2. [Problem Description](#problem-description)
3. [The Base Model](#the-base-model)
4. [Usage Instructions](#run-locally)
6. [Repository Structure](#repository-structure)

## The Dataset
For our tuning process, we have taken a [dataset](https://huggingface.co/datasets/ttbui/html_alpaca) containing about 500 examples where the model is being asked to build an HTML code that solves a given task. Each row contains the description of the task to be solved, an example of data input to the task if applicable, and the generated code fragment that solves the task is provided.

## Problem Description
Our goal is to fine-tune the pre-trained LLaMA-2-7B model's parameters, using 4-bit quantization to produce an HTML coder. This code has been built on Colab T4.

Note: This project is still in progress.

## The Base Model
Meta developed and publicly released the Llama 2 family of large language models (LLMs), a collection of pretrained and fine-tuned generative text models ranging in scale from 7 billion to 70 billion parameters.

Model Architecture Llama 2 is an auto-regressive language model that uses an optimized transformer architecture. The tuned versions use supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF) to align to human preferences for helpfulness and safety

In this project, I've used the [sharded version of LLaMA-2-7B](https://huggingface.co/TinyPixel/Llama-2-7B-bf16-sharded)

## Usage Instructions
The code for the fine-tuning is provided in the attached HTML-Generator.ipynb file. To run it, [open the file](https://github.com/ayucd/HTML-Generator/blob/main/HTML_Generator.ipynb) and click on the Open in Colab button. Further, it's advised that you DO NOT run it on T4 and run it on one of the better GPUs like A100. 

## Repository Structure
This repository is still in construction and more files will be added soon.










