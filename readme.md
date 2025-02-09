# Culturally Sensitive Agent Framework

## Overview

This framework implements a novel approach to creating culturally sensitive AI agents by leveraging Hofstede's Cultural Dimensions Theory. The system generates and evaluates responses that are both culturally appropriate and consistent with the agent's core personality traits.

## Setup

```bash
conda create -n culturally_sensitive_agents
conda activate culturally_sensitive_agents

# Fill out with your own API keys
cp .env.example .env

poetry install

python3 test_cultural_responses.py
```

## Research Context

### Motivation

Current Large Language Models (LLMs) often generate responses that reflect Western cultural biases and may not appropriately adapt to different cultural contexts. This framework addresses this limitation by:

1. Explicitly modeling cultural dimensions using Hofstede's framework
2. Generating responses that respect cultural norms while maintaining character consistency
3. Providing quantitative evaluation metrics for cultural appropriateness

### Theoretical Framework

The system is based on Hofstede's six cultural dimensions:

- **Power Distance Index (PDI)**: Acceptance of power inequality in society
- **Individualism vs. Collectivism (IDV)**: Individual vs. group orientation
- **Masculinity vs. Femininity (MAS)**: Competition vs. cooperation values
- **Uncertainty Avoidance Index (UAI)**: Tolerance for ambiguity and uncertainty
- **Long-term vs. Short-term Orientation (LTO)**: Time horizon for social values
- **Indulgence vs. Restraint (IVR)**: Gratification of basic human desires

## System Architecture

### Components

1. **Cultural Dimension Generator** (`hofstede.py`)
   - Implements core cultural dimension definitions
   - Provides character-culture integration
   - Manages country-specific dimension profiles

2. **Cultural Response Orchestrator** (`cultural_orchestrator.py`)
   - LangGraph-based response generation
   - Cultural appropriateness evaluation
   - Benchmarking and metrics calculation

3. **Testing Framework** (`test_cultural_responses.py`)
   - Example usage and demonstration
   - Results collection and analysis
   - YAML-based output for further analysis

### Key Features

- **Character-Culture Integration**: Maintains character consistency while adapting to cultural norms
- **Quantitative Evaluation**: Scores responses based on cultural appropriateness
- **Extensible Design**: Easy addition of new cultures and character types
- **Reproducible Results**: Structured output format for research analysis

## Usage

### Basic Usage

```python
from pathlib import Path
from cultural_orchestrator import CulturalOrchestrator

# Initialize with character definition
orchestrator = CulturalOrchestrator(Path("character_files/eliza.character.yaml"))

# Generate culturally appropriate responses
responses = orchestrator.generate_responses("japan")

# Evaluate and benchmark
scores = orchestrator.benchmark_responses(responses)
```

### Adding New Cultures

Add new cultural dimensions to `COUNTRY_DIMENSIONS` in `hofstede.py`:

```python
COUNTRY_DIMENSIONS = {
    "new_country": CulturalDimensions(
        power_distance=value,
        individualism=value,
        masculinity=value,
        uncertainty_avoidance=value,
        long_term_orientation=value,
        indulgence=value
    )
}
```

## Research Applications

### Current Research Focus

1. **Cultural Adaptation Quality**
   - Measuring response appropriateness across cultures
   - Analyzing character trait preservation
   - Identifying cultural bias patterns

2. **Cross-Cultural Interaction**
   - Character behavior in multi-cultural contexts
   - Cultural dimension impact on dialogue flow
   - Adaptation strategies effectiveness

3. **Evaluation Metrics**
   - Cultural authenticity scoring
   - Character consistency measurement
   - User acceptance across cultures

### Future Directions

1. **Dynamic Cultural Adaptation**
   - Real-time cultural context switching
   - Multi-cultural interaction modeling
   - Cultural blend handling

2. **Extended Evaluation Framework**
   - Human evaluation integration
   - Cultural expert feedback loop
   - Automated cultural sensitivity detection

3. **Application Areas**
   - Cross-cultural customer service
   - Educational chatbots
   - Cultural sensitivity training

## Contributing

We welcome contributions in the following areas:

1. Additional country dimension profiles
2. Enhanced cultural evaluation metrics
3. New character definitions
4. Improved prompt engineering
5. Documentation and examples

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{culturally_sensitive_agents_2024,
  title={Culturally Sensitive Character Agents: Integrating Hofstede's Dimensions with LLM Response Generation},
  author={[Harrison Dahme]},
  year={2025},
  publisher={GitHub},
  howpublished={\url{https://github.com/[repository-url]}}
}
```

## License

[License Type] - See LICENSE file for details

Compatible with character files from [Eliza](https://github.com/elizaOS/eliza/tree/main/characters)

Convert to yaml with

`cat $CHARACTER_FILE.json | jq '.' | yq eval -P '.' > $CHARACTER_FILE.yaml`

