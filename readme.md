# Evidence-Based RAG Decision System

**基于证据的 RAG 文本决策系统**

---

## Introduction

The **Evidence-Based RAG Decision System** is a document understanding and decision-support system built on **Retrieval-Augmented Generation (RAG)**.

This system is designed to help users **extract key information from documents through questions**, instead of reading the entire document manually.
By asking natural language questions, users can obtain **evidence-backed answers** with references to the original document.

Typical application scenarios include:

* Academic paper reading and understanding
* Key information extraction from long documents
* Evidence-based question answering
* Keyword and concept discovery
* Decision support based on document content

The system currently supports **PDF documents**, and is designed to be extensible to other formats such as **DOCX** in the future.

---

## Developing Environment

This section describes the development environment used for this project.
If you plan to reproduce, modify, or extend the system, the following information may be helpful.

### Hardware

* **Computer Model**: ASUS TUF GAMING FA507RM (华硕天选3，中国版)
* **CPU**: AMD Ryzen 7 6000 Series
* **GPU**: NVIDIA RTX 3060 Laptop (6GB VRAM)
* **Memory**: 16GB DDR4
* **Storage**: 512GB SSD

### Software

* **IDE**: PyCharm 2025.3.1
* **Python**: 3.11
* **LLM Runtime**: Ollama
* **Operating System**: Windows

Detailed software dependencies are described in the **Technique Part**.

---

## Technique Part

First, special thanks to **ChatGPT / OpenAI** for technical assistance during development.
Without these tools, completing this project independently would have been significantly more difficult.

I also want to thank the **NVIDIA RAG-related courses**, which helped me understand the fundamental concepts and practical workflow of Retrieval-Augmented Generation.

This section briefly explains **how the system works**, from document ingestion to answer generation.
Some descriptions may be simplified, and mistakes are possible.

---

## What is RAG?

**Retrieval-Augmented Generation (RAG)** is a technique that combines:

* **Information Retrieval** (finding relevant content from documents)
* **Large Language Models (LLMs)** (generating natural language responses)

Instead of relying solely on the LLM’s internal knowledge, RAG allows the model to:

1. Retrieve relevant document segments
2. Use them as context
3. Generate answers **based on actual evidence**

This approach significantly reduces hallucination and improves answer reliability.

---

## System Architecture Overview

The system follows a modular pipeline design:

1. **Document Loading**

   * Load PDF documents and convert them into structured `Document` objects

2. **Text Chunking**

   * Split documents into semantically meaningful text chunks
   * Preserve metadata such as page numbers and source files

3. **Embedding Generation**

   * Convert text chunks into vector representations using embedding models

4. **Vector Storage**

   * Store embeddings in a vector database (Chroma)

5. **Retrieval**

   * Retrieve the most relevant chunks using similarity search or MMR (Max Marginal Relevance)

6. **Context Construction**

   * Organize retrieved chunks into a structured context with source references

7. **LLM Generation**

   * Generate answers using a Large Language Model based strictly on retrieved evidence

---

## Core Technologies

The system is built using the following technologies:

* **LangChain**
  Used as the orchestration framework connecting loaders, embeddings, retrievers, and LLMs

* **Chroma Vector Database**
  Used for local vector storage and semantic retrieval

* **HuggingFace Embeddings**
  For converting text chunks into dense vector representations

* **Ollama**
  For running local large language models (e.g., Qwen, Phi-3)

* **Python 3.11**
  Main development language

---

## Design Philosophy

This project follows several key design principles:

* **Evidence First**
  Answers must be based on retrieved document content, not free generation

* **Traceability**
  Every answer can be traced back to its original document and page number

* **Modularity**
  Document loaders, embedding models, vector stores, and LLMs are replaceable

* **Extensibility**
  The system is designed to support additional document formats and LLM providers

---

## Current Limitations

* Only PDF documents are supported at the moment
* No graphical user interface (CLI-based usage)
* Limited document structure understanding (e.g., sections, tables)

These limitations are intentional to keep the system lightweight and focused on core RAG functionality.

---

## Future Work

Planned improvements include:

* Support for **DOCX** and other document formats
* Better document structure parsing (sections, headings)
* Multi-document comparison and analysis
* Improved citation formatting and source aggregation
* Web-based user interface
* Integration with cloud-based LLM providers (OpenAI, DeepSeek, NVIDIA NIM)

---


# 基于证据的 RAG 文本决策系统

**Evidence-Based RAG Decision System**

---

## 项目简介

**基于证据的 RAG 文本决策系统** 是一个基于 **检索增强生成（Retrieval-Augmented Generation, RAG）** 的文本理解与决策支持系统。

本系统的目标是：
**通过提问的方式，从文档中获取有证据支撑的关键信息**，而不是依赖模型的自由生成或人工全文阅读。

用户可以使用自然语言向系统提问，系统会基于文档内容给出答案，并且答案**可追溯至原始文档的具体页码或来源**。

典型应用场景包括：

* 学术论文阅读与理解
* 长文档关键信息提取
* 基于证据的问答与决策支持
* 关键词与核心概念发现
* 文档内容快速定位

当前版本主要支持 **PDF 文档**，系统在设计上支持未来扩展至 **DOCX 等其他文档格式**。

---

## 开发环境

本章节介绍项目的开发环境配置，供复现、二次开发或功能扩展参考。

### 硬件环境

* **计算机型号**：ASUS TUF GAMING FA507RM（华硕天选3，中国版）
* **处理器（CPU）**：AMD Ryzen 7 6000 系列
* **显卡（GPU）**：NVIDIA RTX 3060 Laptop（6GB 显存）
* **内存**：16GB DDR4
* **存储**：512GB SSD

### 软件环境

* **开发工具**：PyCharm 2025.3.1
* **Python 版本**：3.11
* **大模型运行环境**：Ollama
* **操作系统**：Windows

更详细的软件依赖信息将在 **技术说明部分** 介绍。

---

## 技术说明

首先感谢 **ChatGPT / OpenAI** 在开发过程中提供的技术支持。
同时感谢 **NVIDIA 的 RAG 相关课程**，帮助我理解了检索增强生成的核心思想和实践流程。

本章节将简要介绍系统的整体运行机制，包括从文档加载到最终答案生成的完整流程。
部分内容为简化说明，如有疏漏，敬请指正。

---

## 什么是 RAG？

**检索增强生成（Retrieval-Augmented Generation, RAG）** 是一种结合了：

* **信息检索（Retrieval）**
* **大语言模型生成（Generation）**

的技术方案。

与传统只依赖模型参数知识的生成方式不同，RAG 的核心思想是：

1. 先从文档中检索与问题相关的内容
2. 将检索到的内容作为上下文
3. 再由大语言模型基于这些证据生成答案

这种方式可以有效降低模型幻觉，提高回答的**准确性与可信度**。

---

## 系统架构概述

系统整体采用模块化流水线设计，主要包括以下步骤：

1. **文档加载（Document Loading）**

   * 读取 PDF 文档并转换为结构化的 `Document` 对象

2. **文本切分（Text Chunking）**

   * 将文档切分为语义完整的文本块
   * 保留页码、来源等元数据信息

3. **向量化（Embedding）**

   * 使用嵌入模型将文本块转换为向量表示

4. **向量存储（Vector Store）**

   * 使用 Chroma 存储向量数据

5. **语义检索（Retrieval）**

   * 基于相似度或 MMR 策略检索相关文本块

6. **上下文构建（Context Construction）**

   * 将检索结果整理为结构化上下文，并保留引用信息

7. **答案生成（Generation）**

   * 使用大语言模型基于检索证据生成最终答案

---

## 核心技术栈

系统主要使用以下技术组件：

* **LangChain**
  用于串联文档加载、向量化、检索与生成流程

* **Chroma 向量数据库**
  用于本地语义向量存储与检索

* **HuggingFace Embeddings**
  用于文本嵌入向量生成

* **Ollama**
  用于本地运行轻量级大语言模型（如 Qwen、Phi-3）

* **Python 3.11**
  系统主要开发语言

---

## 设计理念

本项目遵循以下核心设计原则：

* **证据优先**
  所有回答必须基于文档检索结果，而非模型自由发挥

* **可追溯性**
  每条回答都可以追溯到原始文档及对应页码

* **模块化设计**
  文档加载器、嵌入模型、向量库和大模型均可灵活替换

* **可扩展性**
  系统结构支持未来扩展新的文档格式和模型提供方

---

## 当前限制

* 当前仅支持 PDF 文档
* 尚未提供图形化用户界面（CLI 形式）
* 对文档结构（章节、表格）的理解仍较基础

这些限制是当前阶段为了保持系统简洁和聚焦核心功能而有意保留的。

---

## 未来规划

后续计划包括但不限于：

* 支持 **DOCX 等更多文档格式**
* 更精细的文档结构解析（章节、标题、表格）
* 多文档对比与综合分析
* 更规范的引用与来源聚合机制
* Web 图形界面支持
* 接入云端大模型服务（如 OpenAI、DeepSeek、NVIDIA NIM）

---

