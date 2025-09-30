# Workshop Project: Optimizing FSDP for Function-Calling Fine-tuning

## Executive Summary

This project explores advanced distributed training techniques by fine-tuning Qwen2.5-3B-Instruct for function calling using Fully Sharded Data Parallel (FSDP) on 8xH100 GPUs. The goal is to demonstrate how specialized fine-tuning with optimal distributed strategies can enable a 3B parameter model to compete with much larger models on specific tasks, while providing practical insights for production ML workflows.

## Project Motivation

### Personal Relevance
- **Immediate applicability**: Upcoming work project involving fine-tuning a ~3B parameter model
- **Function calling focus**: Critical capability for production LLM applications (agents, tool use, API integration)
- **Beyond basics**: Moving past simple DDP to explore advanced distributed training optimizations
- **Budget-conscious**: $1,500 budget (~65 hours at $23/hour) requires efficient experimentation

### Technical Significance
Function calling represents a crucial capability for production LLMs, enabling them to:
- Execute API calls from natural language instructions
- Generate properly formatted JSON outputs
- Select appropriate tools from multiple options
- Handle complex multi-step workflows

## Model Selection: Qwen2.5-3B-Instruct

### Why Qwen2.5-3B?
- **Superior structured output**: Excels at JSON generation and formatting compliance
- **Proven tool-use potential**: Qwen3 family demonstrated strong agent capabilities
- **Optimal size**: True 3B parameters fits perfectly within H100 memory constraints
- **Recent architecture**: Benefits from latest training techniques and data

### Baseline Capabilities
- 128K context window support
- Strong instruction following
- Multilingual support (29+ languages)
- Already handles structured data well, providing solid foundation for function-calling specialization

## Dataset Strategy

### Primary Training: Salesforce XLam-60K
- **60,000 high-quality function-calling examples**
- **Proven track record**: Models trained on XLam achieved 88.24% accuracy on BFCL, beating many larger models
- **Diverse scenarios**: Single, multiple, and parallel function calls
- **Clean format**: Well-structured JSON with parameter types and descriptions

### Evaluation: Berkeley Function-Calling Leaderboard (BFCL)
- **Industry standard benchmark** for function calling
- **Multi-language support**: Python, Java, JavaScript, REST APIs
- **Comprehensive categories**:
  - Simple function calls (400 examples)
  - Multiple function selection (200 examples)
  - Parallel execution (200 examples)
  - Irrelevance detection (875 examples)

### Data Distribution Plan
```python
dataset_strategy = {
    "primary_training": {
        "dataset": "Salesforce/xlam-function-calling-60k",
        "examples": 60000,
        "weight": 0.8
    },
    "benchmark_alignment": {
        "dataset": "BFCL training subset",
        "examples": 2000,
        "weight": 0.2
    },
    "validation": {
        "dataset": "BFCL test set",
        "purpose": "Evaluation only"
    }
}
```

## FSDP Implementation Strategy

### Hardware Configuration
- **8x NVIDIA H100 GPUs** (80GB HBM3 each)
- **640GB total VRAM**
- **High-speed NVLink interconnect**

### Memory Optimization
```python
fsdp_configuration = {
    "sharding_strategy": "FULL_SHARD",  # Maximum memory efficiency
    "mixed_precision": {
        "param_dtype": "bfloat16",
        "reduce_dtype": "bfloat16",
        "buffer_dtype": "bfloat16"
    },
    "activation_checkpointing": False,  # Start without, plenty of memory
    "cpu_offload": None,  # Not needed with H100s
}

memory_per_gpu = {
    "model_shards": "0.77 GB",  # 6.18GB / 8
    "optimizer_states": "4.64 GB",  # 37.08GB / 8  
    "gradients": "0.77 GB",  # 6.18GB / 8
    "activations": "~25 GB",  # With batch_size=16
    "total_used": "~31 GB",  # Only 40% of H100 capacity
    "headroom": "49 GB"  # Room for experimentation
}
```

### Training Configuration
```python
training_params = {
    "per_device_batch_size": 16,
    "gradient_accumulation_steps": 4,
    "effective_batch_size": 512,  # 16 * 8 * 4
    "max_sequence_length": 4096,
    "learning_rate": 2e-5,
    "warmup_ratio": 0.1,
    "num_epochs": 3,
    "optimizer": "AdamW",
    "scheduler": "cosine_with_restarts",
    "gradient_clipping": 1.0
}
```

## Experimental Plan

### Phase 1: Baseline & Setup (10 hours, $230)
- **Establish baseline**: Evaluate vanilla Qwen2.5-3B on BFCL
- **Infrastructure validation**: Test FSDP setup, verify memory usage
- **Quick iterations**: Try batch sizes 8, 16, 24, 32
- **Expected baseline**: ~65-70% on BFCL

### Phase 2: Full Fine-tuning (20 hours, $460)
- **Main training run**: 3 epochs on XLam-60K
- **FSDP strategy comparison**:
  - FULL_SHARD vs SHARD_GRAD_OP
  - Impact of activation checkpointing
  - Communication overhead profiling
- **Target performance**: 80-85% on BFCL

### Phase 3: Optimization Experiments (15 hours, $345)
```python
optimization_experiments = {
    "learning_rate_sweep": [1e-5, 2e-5, 5e-5],
    "batch_size_scaling": [256, 512, 1024],
    "sequence_length": [2048, 4096, 8192],
    "mixed_precision": ["fp32", "fp16", "bf16"]
}
```

### Phase 4: LoRA Comparison (10 hours, $230)
- **Parameter-efficient alternatives**:
  - LoRA with ranks [8, 16, 32, 64]
  - QLoRA 4-bit quantization
- **Memory vs performance tradeoffs**
- **Training speed comparison**

### Phase 5: Advanced Techniques (10 hours, $230)
- **Long context experiments**: 8K and 16K sequences
- **Curriculum learning**: Simple → Complex functions
- **Ensemble methods**: Merge multiple LoRA adapters
- **Final benchmark**: Comprehensive BFCL evaluation

## Expected Outcomes

### Performance Targets
```python
expected_results = {
    "baseline_qwen2.5_3b": {
        "bfcl_score": "65-70%",
        "inference_speed": "50 tokens/sec"
    },
    "qwen2.5_3b_xlam_finetuned": {
        "bfcl_score": "80-85%",  # Conservative
        "inference_speed": "45 tokens/sec",
        "improvement": "+15-20%"
    },
    "best_case_scenario": {
        "bfcl_score": "85-90%",  # Optimistic
        "comparable_to": "GPT-3.5-turbo",
        "improvement": "+25-30%"
    }
}
```

### Deliverables
1. **Optimized Training Recipe**: Production-ready FSDP configuration for 3B models
2. **Performance Benchmarks**: Detailed metrics on memory, speed, and accuracy
3. **Cost Analysis**: $/performance ratios for different strategies
4. **Open-source Code**: Complete training pipeline with FSDP optimizations
5. **Model Weights**: Best performing checkpoint on HuggingFace

## Key Innovation Points

### Why This Project Matters
1. **Challenges assumptions**: Demonstrates that specialized 3B models can match 7B+ general models
2. **Practical FSDP insights**: Real-world optimizations beyond textbook examples
3. **Production relevance**: Function calling is critical for enterprise LLM deployments
4. **Efficiency focus**: Maximum performance per dollar/parameter

### Workshop Presentation Value
- **Concrete results**: "Achieved 85% BFCL accuracy with 3B model"
- **Reproducible findings**: Complete configs and code shared
- **Cost-effective approach**: Under $1,500 total investment
- **Immediate applicability**: Attendees can apply techniques to their own projects

## Risk Mitigation

### Potential Challenges
1. **Overfitting**: Mitigate with validation monitoring and early stopping
2. **Catastrophic forgetting**: Use conservative learning rates and LoRA fallback
3. **Format rigidity**: Include diverse function formats in training

### Contingency Plans
- If full fine-tuning fails: Focus on LoRA/QLoRA approaches
- If XLam underperforms: Blend with Glaive/Hermes datasets
- If time runs short: Prioritize best configuration for final runs

## Timeline

| Week | Hours | Cost | Focus |
|------|-------|------|-------|
| 1 | 20 | $460 | Baseline + Initial FSDP experiments |
| 2 | 25 | $575 | Full training + Optimization sweeps |
| 3 | 20 | $460 | LoRA comparison + Advanced techniques |
| **Total** | **65** | **$1,495** | **Complete project within budget** |

## Success Criteria

✅ **Technical Success**: Beat baseline Qwen2.5-3B by >15% on BFCL  
✅ **Learning Success**: Master FSDP optimizations applicable to production  
✅ **Workshop Success**: Deliver compelling presentation with reproducible results  
✅ **Practical Success**: Create reusable training pipeline for future work  

## Conclusion

This project combines cutting-edge distributed training techniques with a practical, high-impact use case. By fine-tuning Qwen2.5-3B for function calling using FSDP optimizations, we'll demonstrate that thoughtful engineering and specialized training can enable smaller models to punch above their weight class. The insights gained will be immediately applicable to production ML workflows while advancing the state of open-source function-calling models.


## References:

- https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k
