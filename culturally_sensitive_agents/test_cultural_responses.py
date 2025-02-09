"""
Test script for demonstrating cultural prompt adaptation and evaluation.
"""
import traceback
from pathlib import Path
import yaml
import logging
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from cultural_orchestrator import CulturalOrchestrator
from hofstede import PromptAdaptation

import dotenv
dotenv.load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Setup rich console
console = Console()

def print_adaptation_table(country: str, adaptations: list, metrics: dict, use_character_tone: bool = False):
    """Print prompt adaptations in a formatted table."""
    # Print iteration info if available
    if "iteration" in metrics:
        console.print(f"\n[bold cyan]Iteration {metrics['iteration']}[/bold cyan]")
    
    # Print original vs adapted prompts
    table = Table(title=f"Prompt Adaptations for {country.upper()}")
    
    # Add columns
    table.add_column("Dimension", style="cyan")
    table.add_column("Original", style="yellow")
    table.add_column("Adapted", style="green")
    table.add_column("Cultural Score", justify="right", style="magenta")
    if use_character_tone:
        table.add_column("Consistency", justify="right", style="blue")
    table.add_column("Improvement", justify="right", style="red")
    
    # Add rows
    for adaptation in adaptations:
        # Calculate improvement percentage using actual baseline
        improvement = adaptation.calculate_improvement()
        row = [
            adaptation.target_dimension,
            adaptation.original_prompt,
            adaptation.adapted_prompt,
            f"{adaptation.cultural_acceptance_score:.2f} (base: {adaptation.original_cultural_score:.2f})",
        ]
        if use_character_tone:
            row.append(f"{adaptation.character_consistency_score:.2f} (base: {adaptation.original_consistency_score:.2f})")
        row.append(f"{improvement:+.1f}%" if improvement != 0 else "0%")
        table.add_row(*row)
    
    console.print(table)
    
    # Print improvement metrics
    console.print("\n[bold]Improvement Metrics[/bold]")
    improvements = metrics["average_improvement"]
    avg_baseline = sum(a.original_cultural_score for a in adaptations) / len(adaptations)
    avg_improvement = ((improvements["cultural_acceptance"] - avg_baseline) / avg_baseline) * 100 if avg_baseline > 0 else 0
    
    metrics_text = [
        ("Average Cultural Acceptance: ", "bold"),
        f"{improvements['cultural_acceptance']:.2f} (baseline: {avg_baseline:.2f}, {avg_improvement:+.1f}%)\n",
    ]
    if use_character_tone:
        metrics_text.extend([
            ("Character Consistency: ", "bold"),
            f"{improvements['character_consistency']:.2f}"
        ])
    
    console.print(Panel(
        Text.assemble(*metrics_text),
        title="Overall Improvements"
    ))
    
    # Print dimension-specific improvements
    console.print("\n[bold]Dimension-specific Improvements[/bold]")
    dim_table = Table()
    dim_table.add_column("Dimension", style="cyan")
    dim_table.add_column("Cultural Score", justify="right", style="magenta")
    dim_table.add_column("Baseline", justify="right", style="yellow")
    if use_character_tone:
        dim_table.add_column("Consistency", justify="right", style="blue")
    dim_table.add_column("Improvement", justify="right", style="red")
    
    for dim, scores in metrics["dimension_improvements"].items():
        # Find the adaptation for this dimension
        dim_adaptation = next((a for a in adaptations if a.target_dimension == dim), None)
        if dim_adaptation:
            improvement = dim_adaptation.calculate_improvement()
            row = [
                dim,
                f"{scores['cultural_acceptance']:.2f}",
                f"{dim_adaptation.original_cultural_score:.2f}",
            ]
            if use_character_tone:
                row.append(f"{scores['character_consistency']:.2f}")
            row.append(f"{improvement:+.1f}%")
            dim_table.add_row(*row)
    
    console.print(dim_table)

def print_before_after_comparison(country: str, original_prompt: str, best_adaptation: PromptAdaptation, use_character_tone: bool = False):
    """Print a before/after comparison of the prompt adaptation."""
    console.print("\n[bold cyan]Before/After Comparison[/bold cyan]")
    
    # Create comparison table for prompts and responses
    table = Table(title=f"Final Adaptation Result for {country.upper()}")
    table.add_column("Version", style="cyan", width=12)
    table.add_column("Response", style="green")
    table.add_column("Cultural Score", justify="right", style="magenta", width=12)
    if use_character_tone:
        table.add_column("Character Voice", style="blue")
    
    # Add original version with actual baseline score
    row = [
        "Original",
        best_adaptation.original_response,
        f"{best_adaptation.original_cultural_score:.2f}",
    ]
    if use_character_tone:
        row.append("Baseline character voice")
    table.add_row(*row)
    
    # Add adapted version
    row = [
        "Adapted",
        best_adaptation.adapted_response,
        f"{best_adaptation.cultural_acceptance_score:.2f}",
    ]
    
    if use_character_tone:
        # Get character analysis from evaluation results
        character_analysis = {
            "tone_preservation": 0.0,
            "personality_match": 0.0,
            "voice_authenticity": 0.0,
            "comments": "No character analysis available"
        }
        
        if best_adaptation.evaluation_results:
            latest_eval = best_adaptation.evaluation_results[-1]
            if "character_analysis" in latest_eval:
                character_analysis = latest_eval["character_analysis"]
        
        # Calculate voice score
        voice_score = (
            character_analysis["tone_preservation"] +
            character_analysis["personality_match"] +
            character_analysis["voice_authenticity"]
        ) / 3
        
        voice_analysis = f"Voice Score: {voice_score:.2f}\n{character_analysis['comments']}"
        row.append(voice_analysis)
    
    table.add_row(*row)
    console.print(table)
    
    # Print adaptation details
    details = [
        ("Target Dimension: ", "bold"),
        f"{best_adaptation.target_dimension}\n",
        ("Adaptation Explanation: ", "bold"),
        f"{best_adaptation.adaptation_explanation}\n",
        ("Cultural Improvement: ", "bold"),
        f"{best_adaptation.calculate_improvement():+.1f}%\n",
    ]
    
    if use_character_tone:
        details.extend([
            "\n",
            ("Character Voice Analysis: ", "bold red"),
            f"\n• Tone Preservation: {character_analysis['tone_preservation']:.2f}",
            f"\n• Personality Match: {character_analysis['personality_match']:.2f}",
            f"\n• Voice Authenticity: {character_analysis['voice_authenticity']:.2f}",
        ])
    
    details.extend([
        "\n\nKey Differences:",
        "\n• Original response shows baseline cultural awareness",
        "\n• Adapted response enhances cultural sensitivity",
    ])
    
    if use_character_tone:
        details.extend([
            "\n• Shows cultural adaptation while preserving personality",
            "\n• Demonstrates authentic character voice in cultural context"
        ])
    
    console.print(Panel(
        Text.assemble(*details),
        title="Adaptation Details"
    ))

def save_research_outputs(results: dict, base_output_dir: Path):
    """Save detailed results for research purposes."""
    # Create output directories
    base_output_dir.mkdir(parents=True, exist_ok=True)
    character_dir = base_output_dir / "character_adaptations"
    character_dir.mkdir(exist_ok=True)
    
    # Save raw results
    with open(base_output_dir / "cultural_adaptation_results.yaml", "w") as f:
        yaml.dump(results, f, default_flow_style=False)
    
    # Extract and save character cultural suggestions
    character_suggestions = {}
    for prompt in results:
        for country, country_data in results[prompt].items():
            if country not in character_suggestions:
                character_suggestions[country] = []
            
            # Extract cultural suggestions from adaptations
            for adaptation in country_data["adaptations"]:
                if "evaluation_results" in adaptation:
                    for eval_result in adaptation["evaluation_results"]:
                        if "cultural_suggestions" in eval_result:
                            suggestion = eval_result["cultural_suggestions"]
                            if suggestion not in character_suggestions[country]:
                                character_suggestions[country].append(suggestion)
    
    # Save character cultural suggestions
    for country, suggestions in character_suggestions.items():
        with open(character_dir / f"{country}_character_adaptations.yaml", "w") as f:
            yaml.dump({
                "country": country,
                "cultural_adaptations": suggestions
            }, f, default_flow_style=False)
    
    # Generate research summary
    summary = {
        "test_cases": [],
        "overall_metrics": {
            "total_cases": 0,
            "total_iterations": 0,
            "avg_iterations": 0,
            "avg_improvement": 0,
            "cultural_acceptance": {
                "min": float('inf'),
                "max": float('-inf'),
                "avg": 0
            },
            "character_consistency": {
                "min": float('inf'),
                "max": float('-inf'),
                "avg": 0
            }
        },
        "country_specific": {},
        "dimension_analysis": {},
        "character_adaptations": character_suggestions
    }
    
    # Process results
    total_scores = 0
    for prompt in results:
        case_summary = {
            "prompt": prompt,
            "countries": {}
        }
        
        for country in results[prompt]:
            country_data = results[prompt][country]
            metrics = country_data["metrics"]
            
            # Update country-specific stats
            if country not in summary["country_specific"]:
                summary["country_specific"][country] = {
                    "total_cases": 0,
                    "avg_cultural_acceptance": 0,
                    "avg_character_consistency": 0,
                    "avg_improvement": 0,
                    "avg_baseline": 0  # Add baseline tracking
                }
            
            country_stats = summary["country_specific"][country]
            country_stats["total_cases"] += 1
            
            # Calculate average baseline for this country's adaptations
            country_baseline = sum(
                adaptation["original_cultural_score"]
                for adaptation in country_data["adaptations"]
            ) / len(country_data["adaptations"])
            
            # Update running averages
            country_stats["avg_baseline"] += country_baseline
            country_stats["avg_cultural_acceptance"] += metrics["average_improvement"]["cultural_acceptance"]
            country_stats["avg_character_consistency"] += metrics["average_improvement"]["character_consistency"]
            
            # Update case summary
            case_summary["countries"][country] = {
                "iterations": country_data["iterations_needed"],
                "cultural_acceptance": metrics["average_improvement"]["cultural_acceptance"],
                "character_consistency": metrics["average_improvement"]["character_consistency"],
                "baseline_score": country_baseline,
                "dimension_improvements": metrics["dimension_improvements"]
            }
            
            # Update overall metrics
            summary["overall_metrics"]["total_cases"] += 1
            summary["overall_metrics"]["total_iterations"] += country_data["iterations_needed"]
            
            # Update min/max scores
            ca_score = metrics["average_improvement"]["cultural_acceptance"]
            cc_score = metrics["average_improvement"]["character_consistency"]
            
            summary["overall_metrics"]["cultural_acceptance"]["min"] = min(
                summary["overall_metrics"]["cultural_acceptance"]["min"],
                ca_score
            )
            summary["overall_metrics"]["cultural_acceptance"]["max"] = max(
                summary["overall_metrics"]["cultural_acceptance"]["max"],
                ca_score
            )
            summary["overall_metrics"]["character_consistency"]["min"] = min(
                summary["overall_metrics"]["character_consistency"]["min"],
                cc_score
            )
            summary["overall_metrics"]["character_consistency"]["max"] = max(
                summary["overall_metrics"]["character_consistency"]["max"],
                cc_score
            )
            
            # Accumulate for averages
            summary["overall_metrics"]["cultural_acceptance"]["avg"] += ca_score
            summary["overall_metrics"]["character_consistency"]["avg"] += cc_score
            
            # Process dimension improvements
            for dim, scores in metrics["dimension_improvements"].items():
                if dim not in summary["dimension_analysis"]:
                    summary["dimension_analysis"][dim] = {
                        "total_improvements": 0,
                        "avg_cultural_acceptance": 0,
                        "avg_character_consistency": 0
                    }
                
                dim_stats = summary["dimension_analysis"][dim]
                dim_stats["total_improvements"] += 1
                dim_stats["avg_cultural_acceptance"] += scores["cultural_acceptance"]
                dim_stats["avg_character_consistency"] += scores["character_consistency"]
            
            total_scores += 1
        
        summary["test_cases"].append(case_summary)
    
    # Calculate averages
    if total_scores > 0:
        summary["overall_metrics"]["avg_iterations"] = summary["overall_metrics"]["total_iterations"] / total_scores
        summary["overall_metrics"]["cultural_acceptance"]["avg"] /= total_scores
        summary["overall_metrics"]["character_consistency"]["avg"] /= total_scores
        
        # Calculate country averages
        for country in summary["country_specific"]:
            country_stats = summary["country_specific"][country]
            total = country_stats["total_cases"]
            if total > 0:
                country_stats["avg_baseline"] /= total
                country_stats["avg_cultural_acceptance"] /= total
                country_stats["avg_character_consistency"] /= total
                # Calculate improvement using actual baseline
                if country_stats["avg_baseline"] > 0:
                    country_stats["avg_improvement"] = (
                        (country_stats["avg_cultural_acceptance"] - country_stats["avg_baseline"]) 
                        / country_stats["avg_baseline"]
                    ) * 100
                else:
                    country_stats["avg_improvement"] = 0
        
        # Calculate dimension averages
        for dim in summary["dimension_analysis"]:
            dim_stats = summary["dimension_analysis"][dim]
            total = dim_stats["total_improvements"]
            if total > 0:
                dim_stats["avg_cultural_acceptance"] /= total
                dim_stats["avg_character_consistency"] /= total
    
    # Save research summary
    with open(base_output_dir / "research_summary.yaml", "w") as f:
        yaml.dump(summary, f, default_flow_style=False)
    
    return summary

def main():
    """Run the cultural prompt adaptation and evaluation demo."""
    use_character_tone = False  # Control whether to use character tone
    
    # Initialize the orchestrator
    orchestrator_adapted = CulturalOrchestrator(
        character_file=Path("character_files/eliza.character.yaml"),
        acceptance_threshold=0.7,  # Higher threshold for better quality
        max_iterations=3,
        use_character_tone=use_character_tone
    )
    logger.info(f"Initialized Cultural Orchestrator with character tone {'enabled' if use_character_tone else 'disabled'}")
    
    # Load character definition only if using character tone
    character_def = None
    if use_character_tone:
        with open(Path("character_files/eliza.character.yaml"), 'r') as f:
            character_def = yaml.safe_load(f)
    
    # Test prompts and countries
    test_cases = [
        {
            "prompt": "How should I approach my team about a new project?",
            "countries": ["japan", "usa", "china", "brazil"]
        },
        {
            "prompt": "What's the best way to give feedback to a colleague?",
            "countries": ["china", "india", "russia"]
        },
        {
            "prompt": "How do I negotiate a salary increase?",
            "countries": ["uae", "usa", "south_korea"]
        }
    ]
    
    # Generate and evaluate adaptations for each case
    results = {}
    for case in test_cases:
        console.rule(f"[bold blue]Testing Prompt: {case['prompt']}")
        logger.info(f"Processing prompt: {case['prompt']}")
        
        prompt_results = {}
        for country in case["countries"]:
            console.print(f"\n[bold]Adapting for {country.upper()}[/bold]")
            logger.info(f"Processing country: {country}")
            
            try:
                # Generate and evaluate adaptations
                adaptations, metrics = orchestrator_adapted.adapt_and_evaluate(country, case["prompt"])
                
                # Store results
                prompt_results[country] = {
                    "adaptations": [
                        {
                            **a.__dict__,
                            "evaluation_results": metrics.get("evaluation_results", [])
                        } for a in adaptations
                    ],
                    "metrics": metrics,
                    "iterations_needed": metrics["iteration"]
                }
                
                # Print results with comparison
                print_adaptation_table(country, adaptations, metrics, use_character_tone)
                
                # Print summary if multiple iterations were needed
                if metrics["iteration"] > 1:
                    console.print(f"\n[bold yellow]Required {metrics['iteration']} iterations to reach acceptance threshold[/bold yellow]")
                
                # Find the best adaptation based on cultural acceptance score
                best_adaptation = max(
                    adaptations,
                    key=lambda x: x.cultural_acceptance_score
                )
                
                # Show before/after comparison
                print_before_after_comparison(country, case["prompt"], best_adaptation, use_character_tone)
                
            except Exception as e:
                logger.error(f"Error processing {country}: {str(e)}")
                logger.error(traceback.format_exc())
                continue
            
            # Add a separator between countries
            console.print("\n" + "="*80 + "\n")
        
        results[case["prompt"]] = prompt_results
        
        # Add a separator between test cases
        console.print("\n" + "#"*80 + "\n")
    
    # Save detailed results for research
    output_dir = Path("outputs/cultural_adaptation_study")
    character_dir = output_dir / "character_adaptations"  # Define character_dir here
    summary = save_research_outputs(results, output_dir)
    
    # Print final summary
    console.print("\n[bold green]Overall Statistics[/bold green]")
    console.print(f"Average iterations per case: {summary['overall_metrics']['avg_iterations']:.2f}")
    console.print(f"Total test cases: {summary['overall_metrics']['total_cases']}")
    
    console.print("\n[bold]Cultural Acceptance Scores[/bold]")
    ca_stats = summary["overall_metrics"]["cultural_acceptance"]
    console.print(f"Min: {ca_stats['min']:.2f}")
    console.print(f"Max: {ca_stats['max']:.2f}")
    console.print(f"Avg: {ca_stats['avg']:.2f}")
    
    console.print("\n[bold]Per-Country Improvements[/bold]")
    for country, stats in summary["country_specific"].items():
        console.print(
            f"{country.upper()}: {stats['avg_cultural_acceptance']:.2f} "
            f"(baseline: {stats['avg_baseline']:.2f}, "
            f"improvement: {stats['avg_improvement']:+.1f}%)"
        )
    
    console.print("\n[bold]Character Cultural Adaptations[/bold]")
    console.print("Saved to:", character_dir)
    for country in summary["character_adaptations"]:
        console.print(f"\n{country.upper()}:")
        console.print(f"- {character_dir / f'{country}_character_adaptations.yaml'}")
    
    console.print("\n[bold]Results saved to:[/bold]", output_dir)

if __name__ == "__main__":
    main() 