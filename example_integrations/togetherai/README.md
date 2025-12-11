# Arcee AI Trinity-Mini Voice Agent Demo

A real-time customer service voice agent showcasing **Arcee AI's Trinity-Mini** on **Together AI** with **Cartesia Voice Agents**. This demo demonstrates how to build intelligent voice applications using one of the most cost-efficient reasoning models available.

## About Trinity-Mini

[Trinity-Mini](https://huggingface.co/arcee-ai/trinity-mini) is Arcee AI's compact reasoning model, part of the Trinity family—a serious open-weight model family trained end-to-end in the United States.

### Key Highlights

- **Architecture**: 26B parameter MoE (Mixture of Experts) with only 3B active parameters per token
- **Designed for Agents**: Optimized for tools, function calling, and multi-step reasoning workloads
- **US-Trained**: Full end-to-end training in America with clean data provenance
- **Open Weights**: Released under Apache 2.0 license
- **Cost Efficient**: One of the most affordable models at $0.045/$0.15 per million tokens on Together AI

### Technical Architecture

Trinity-Mini incorporates cutting-edge techniques:
- **Gated Attention**: G1 configuration for learned modulation of attention outputs
- **DeepSeekMoE Design**: 128 routed experts with 8 active per token, plus 1 shared expert
- **Sigmoid Routing**: Aux-loss-free load balancing for cleaner training objectives
- **Extended Context**: Trained at 128k sequence length with local/global attention patterns

## Demo Architecture

```
┌─────────────────┐    ┌─────────────────┐
│   Customer      │    │ Escalation      │
│   Service Node  │    │ Monitor Node    │
│ (Trinity-Mini)  │    │ (Trinity-Mini)  │
├─────────────────┤    ├─────────────────┤
│ • Knowledge Base│    │ • Frustration   │
│ • Ticket System │    │   Detection     │
│ • Human Handoff │    │ • Complexity    │
│ • Issue Routing │    │   Assessment    │
└─────────────────┘    └─────────────────┘
         │                       │
         └───────────────────────┘
                    │
            ┌───────────────┐
            │ Cartesia Line │
            │ Voice System  │
            └───────────────┘
```

This example uses Trinity-Mini for both the main conversational agent and the background escalation monitor, demonstrating its versatility for multi-agent voice applications.

## Features

### **Customer Service Capabilities**
- **Knowledge Base Search**: Automated lookup of common solutions
- **Support Ticket Creation**: Generate tickets for complex issues
- **Human Escalation**: Intelligent handoff to human agents
- **Issue Classification**: Automatic routing based on problem type

### **Background Monitoring**
- **Escalation Detection**: Monitors conversation for frustration signals
- **Complexity Assessment**: Identifies issues requiring human intervention
- **Real-time Analysis**: Continuous conversation monitoring with structured outputs

### **Customer Support Tools**
- `search_knowledge_base`: Query FAQ and documentation
- `create_ticket`: Generate support tickets with priorities
- `escalate_to_human`: Transfer to human agents with context
- `end_call`: Graceful conversation termination

## Getting Started

### Prerequisites
- [Together AI API Key](https://api.together.xyz/settings/api-keys)
- [Cartesia Account](https://play.cartesia.ai/agents) and API key

### Setup

1. **Install Dependencies**
   ```bash
   pip install cartesia-line openai python-dotenv loguru aiohttp uvicorn
   ```

2. **Environment Configuration**
   Create a `.env` file or add to your Cartesia account:
   ```
   TOGETHER_API_KEY=your_together_ai_api_key_here
   ```

3. **Configuration**
   - System prompts and model settings in `config.py`
   - Knowledge base entries can be customized
   - Escalation triggers are configurable

## Implementation Details

### **Main Components**

**CustomerServiceNode** (`customer_service_node.py`)
- Primary conversational agent powered by Trinity-Mini
- Handles customer interactions and tool execution
- Manages knowledge base searches and ticket creation
- Coordinates human escalation when needed

**EscalationNode** (`escalation_node.py`)
- Background monitoring using Trinity-Mini with structured JSON outputs
- Analyzes conversation patterns for escalation triggers
- Provides structured escalation recommendations (LOW/MEDIUM/HIGH)
- Tracks escalation history and patterns

**Utility Functions** (`openai_utils.py`)
- Message format conversion for Together AI's OpenAI-compatible API
- Mock implementations of customer service backends
- Tool schema definitions for function calling
- Helper functions for ticket creation and knowledge base search

### **Model Configuration**

Trinity-Mini uses these recommended settings (configured in `config.py`):
```python
TEMPERATURE = 0.15
TOP_P = 0.75
```

### **Customer Service Workflow**

1. **Initial Greeting**: Welcome customer and identify issue type
2. **Issue Assessment**: Classify the problem and search knowledge base
3. **Resolution Attempt**: Provide solutions using available tools
4. **Escalation Decision**: Monitor for complexity or frustration
5. **Human Handoff**: Transfer with context when escalation needed
6. **Ticket Creation**: Generate support tickets for follow-up

### **Escalation Triggers**
- Customer expresses frustration multiple times
- Issue remains unresolved after several attempts
- Explicit request for human assistance
- Technical complexity beyond automation
- Account security or billing concerns

## Deployment

### **Local Testing**
```bash
python main.py
```

### **Cartesia Platform**
1. Add this directory to your [Agents Dashboard](https://play.cartesia.ai/agents)
2. Configure environment variables in the platform
3. Deploy and test with voice interactions

### **Configuration Files**
- `pyproject.toml`: Dependencies and project metadata
- `config.py`: Prompts, models, and behavioral settings
- `.env`: API keys and sensitive configuration

## Customization

### **Knowledge Base**
Edit `KNOWLEDGE_BASE` in `config.py` to add company-specific information:
```python
KNOWLEDGE_BASE = {
    "login": "Your login instructions...",
    "billing": "Your billing process...",
    "technical": "Your technical support steps...",
}
```

### **Escalation Criteria**
Modify `prompt_escalation` in `config.py` to adjust escalation sensitivity:
- Frustration thresholds
- Complexity indicators
- Keywords for automatic escalation

### **Customer Service Tools**
Extend `openai_utils.py` to add new capabilities:
- CRM system integration
- Live chat transfer
- Email support routing
- Analytics and reporting

## Example Interactions

**Knowledge Base Query:**
- Customer: "I can't log in to my account"
- Agent: Searches knowledge base → Provides password reset instructions

**Ticket Creation:**
- Customer: "My software keeps crashing with error code 503"
- Agent: Creates high-priority technical support ticket

**Human Escalation:**
- Customer: "This is the third time calling about this billing issue"
- System: Detects escalation need → Transfers to human agent

## Integration Patterns

This example demonstrates key patterns for Line SDK integrations:
- **Multi-node Architecture**: Primary + background processing with the same model
- **Tool Integration**: OpenAI-compatible function calling with custom tools
- **Event-driven Communication**: Bridge system for inter-node messaging
- **Structured Outputs**: JSON schemas for consistent escalation analysis
- **Error Handling**: Graceful degradation and user-friendly error messages

## Learn More

- **Trinity-Mini on Hugging Face**: [arcee-ai/trinity-mini](https://huggingface.co/arcee-ai/trinity-mini)
- **Arcee AI**: [arcee.ai](https://arcee.ai)
- **Together AI**: [together.ai](https://together.ai)
- **Cartesia Line Docs**: [docs.cartesia.ai/line](https://docs.cartesia.ai/line/introduction)
- **Community**: [Discord](https://discord.gg/cartesia)
