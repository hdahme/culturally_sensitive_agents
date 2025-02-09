"""
Hofstede's Cultural Dimensions implementation for culturally sensitive prompt adaptation.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import yaml
from pathlib import Path

@dataclass
class CulturalDimensions:
    """Represents Hofstede's six cultural dimensions for a specific culture."""
    power_distance: float  # PDI: Power Distance Index
    individualism: float   # IDV: Individualism vs. Collectivism
    masculinity: float     # MAS: Masculinity vs. Femininity
    uncertainty_avoidance: float  # UAI: Uncertainty Avoidance Index
    long_term_orientation: float  # LTO: Long-term vs. Short-term Orientation
    indulgence: float     # IVR: Indulgence vs. Restraint
    
    def to_dict(self) -> Dict[str, float]:
        """Convert dimensions to dictionary format."""
        return {
            "power_distance": self.power_distance,
            "individualism": self.individualism,
            "masculinity": self.masculinity,
            "uncertainty_avoidance": self.uncertainty_avoidance,
            "long_term_orientation": self.long_term_orientation,
            "indulgence": self.indulgence
        }
    
    def get_dominant_dimensions(self, threshold: float = 70.0) -> List[str]:
        """Get the dominant cultural dimensions (those above threshold)."""
        dominant = []
        for dim, value in self.to_dict().items():
            if value > threshold:
                dominant.append(dim)
        return dominant
    
    def get_dimension_guidelines(self) -> Dict[str, Tuple[str, str]]:
        """Get adaptation guidelines for each dimension."""
        return {
            "power_distance": (
                "Show respect for hierarchy and authority" if self.power_distance > 50
                else "Emphasize equality and flat relationships"
            ),
            "individualism": (
                "Focus on individual achievement and personal goals" if self.individualism > 50
                else "Emphasize group harmony and collective success"
            ),
            "masculinity": (
                "Be direct and focus on achievement" if self.masculinity > 50
                else "Focus on consensus and quality of relationships"
            ),
            "uncertainty_avoidance": (
                "Provide clear structure and detailed plans" if self.uncertainty_avoidance > 50
                else "Allow for flexibility and ambiguity"
            ),
            "long_term_orientation": (
                "Emphasize future benefits and growth" if self.long_term_orientation > 50
                else "Focus on immediate results and traditions"
            ),
            "indulgence": (
                "Encourage enjoyment and flexibility" if self.indulgence > 50
                else "Maintain restraint and follow norms"
            )
        }

@dataclass
class PromptAdaptation:
    """Represents a culturally adapted prompt with its context."""
    original_prompt: str
    adapted_prompt: str
    target_dimension: str
    adaptation_explanation: str
    cultural_acceptance_score: float
    character_consistency_score: float
    original_response: str = ""  # Response to original prompt
    adapted_response: str = ""   # Response to adapted prompt
    original_cultural_score: float = 0.0  # Cultural score for original response
    original_consistency_score: float = 0.0  # Character consistency for original response
    evaluation_results: List[Dict[str, Any]] = None  # Detailed evaluation results

    def __post_init__(self):
        """Initialize empty list for evaluation results if None."""
        if self.evaluation_results is None:
            self.evaluation_results = []
            
    def calculate_improvement(self) -> float:
        """Calculate improvement percentage over original response."""
        if self.original_cultural_score > 0:
            return ((self.cultural_acceptance_score - self.original_cultural_score) / self.original_cultural_score) * 100
        return 0.0

class CultureAwareAdapter:
    """Adapts prompts to be culturally appropriate while maintaining character consistency."""
    
    def __init__(
        self, 
        character_file: Path,
        dimensions: CulturalDimensions,
        name: Optional[str] = None,
        use_character_tone: bool = False
    ):
        self.dimensions = dimensions
        self.name = name
        self.use_character_tone = use_character_tone
        self.character_data = self._load_character(character_file) if use_character_tone else {}
        
    def _load_character(self, character_file: Path) -> dict:
        """Load character definition from YAML file."""
        if not self.use_character_tone:
            return {}
            
        with open(character_file, 'r') as f:
            return yaml.safe_load(f)
    
    def _get_character_tone(self) -> str:
        """Extract character's tone and personality traits.
        
        Returns:
            str: A comprehensive description of the character's tone, combining style,
                personality traits, and voice characteristics.
        """
        if not self.use_character_tone:
            return "helpful and professional"
        
        # Initialize trait categories
        style_traits = []
        personality_traits = []
        voice_traits = []
        tone_traits = []
        
        # Extract style guidelines
        if "style" in self.character_data:
            style = self.character_data["style"]
            if "all" in style:
                style_traits.extend(style["all"])
            if "chat" in style:
                style_traits.extend(style["chat"])
        
        # Extract personality traits
        if "personality" in self.character_data:
            personality = self.character_data["personality"]
            if "traits" in personality:
                personality_traits.extend(personality["traits"])
            if "values" in personality:
                personality_traits.extend(personality["values"])
        
        # Extract voice characteristics
        if "voice" in self.character_data:
            voice = self.character_data["voice"]
            if "style" in voice:
                voice_traits.extend(voice["style"])
            if "characteristics" in voice:
                voice_traits.extend(voice["characteristics"])
        
        # Extract tone specifications
        if "tone" in self.character_data:
            tone = self.character_data["tone"]
            if "characteristics" in tone:
                tone_traits.extend(tone["characteristics"])
            if "style" in tone:
                tone_traits.extend(tone["style"])
        
        # Combine traits with appropriate weighting
        all_traits = []
        
        # Style guidelines are most important for cultural adaptation
        if style_traits:
            all_traits.append("style: " + ", ".join(style_traits))
        
        # Personality traits help maintain character consistency
        if personality_traits:
            all_traits.append("personality: " + ", ".join(personality_traits))
        
        # Voice and tone traits provide specific guidance
        if voice_traits:
            all_traits.append("voice: " + ", ".join(voice_traits))
        if tone_traits:
            all_traits.append("tone: " + ", ".join(tone_traits))
        
        # Get character's system rules
        system_rules = self.character_data.get("system", "helpful and professional")
        
        # Combine everything into a comprehensive tone description
        if all_traits:
            return "; ".join(all_traits) + f"; following system rules: {system_rules}"
        else:
            return "helpful and professional while " + system_rules
    
    def adapt_prompt(self, original_prompt: str) -> List[PromptAdaptation]:
        """Generate culturally adapted versions of the original prompt."""
        adaptations = []
        guidelines = self.dimensions.get_dimension_guidelines()
        
        for dimension, guideline in guidelines.items():
            # Create adapted prompt based on cultural dimension
            adapted_prompt = self._create_adapted_prompt(
                original_prompt, 
                dimension, 
                guideline
            )
            
            # Create adaptation object with initial scores
            adaptation = PromptAdaptation(
                original_prompt=original_prompt,
                adapted_prompt=adapted_prompt,
                target_dimension=dimension,
                adaptation_explanation=f"Adapted for {dimension}: {guideline}",
                cultural_acceptance_score=0.0,  # Will be filled by evaluator
                character_consistency_score=0.0  # Will be filled by evaluator
            )
            adaptations.append(adaptation)
            
        return adaptations
    
    def adapt_prompt_with_feedback(self, original_prompt: str, feedback: Dict[str, Any]) -> List[PromptAdaptation]:
        """Generate culturally adapted versions of the prompt using feedback."""
        adaptations = []
        guidelines = self.dimensions.get_dimension_guidelines()
        
        # Focus on priority dimensions from feedback
        priority_dimensions = feedback.get("priority_dimensions", [])
        dimension_feedback = feedback.get("dimension_feedback", {})
        
        for dimension, guideline in guidelines.items():
            # Get dimension-specific feedback if available
            dim_feedback = dimension_feedback.get(dimension, {})
            suggestions = dim_feedback.get("suggestions", [])
            
            # Create adapted prompt with feedback incorporation
            adapted_prompt = self._create_adapted_prompt_with_feedback(
                original_prompt,
                dimension,
                guideline,
                suggestions,
                is_priority=dimension in priority_dimensions
            )
            
            # Create adaptation object
            adaptation = PromptAdaptation(
                original_prompt=original_prompt,
                adapted_prompt=adapted_prompt,
                target_dimension=dimension,
                adaptation_explanation=f"Adapted for {dimension} with feedback: {', '.join(suggestions)}",
                cultural_acceptance_score=0.0,
                character_consistency_score=0.0
            )
            adaptations.append(adaptation)
        
        return adaptations
    
    def _create_adapted_prompt(self, original_prompt: str, dimension: str, guideline: str) -> str:
        """Create a culturally adapted version of the prompt."""
        # Add cultural context based on dimension
        cultural_context = {
            "power_distance": "Considering the organizational hierarchy",
            "individualism": "Thinking about personal/group dynamics",
            "masculinity": "Focusing on achievement and relationships",
            "uncertainty_avoidance": "Regarding structure and planning",
            "long_term_orientation": "Considering future implications",
            "indulgence": "Balancing enjoyment and restraint"
        }
        
        # Combine original prompt with cultural adaptation
        adapted_prompt = f"{cultural_context[dimension]}, {guideline}: {original_prompt}"
        
        # Add character tone only if enabled
        if self.use_character_tone:
            tone = self._get_character_tone()
            adapted_prompt = f"Using a tone that is {tone}, respond to: {adapted_prompt}"
            
        return adapted_prompt
    
    def _create_adapted_prompt_with_feedback(
        self,
        original_prompt: str,
        dimension: str,
        guideline: str,
        suggestions: List[str],
        is_priority: bool
    ) -> str:
        """Create a culturally adapted version of the prompt incorporating feedback."""
        # Get base cultural context
        cultural_context = {
            "power_distance": "Considering the organizational hierarchy",
            "individualism": "Thinking about personal/group dynamics",
            "masculinity": "Focusing on achievement and relationships",
            "uncertainty_avoidance": "Regarding structure and planning",
            "long_term_orientation": "Considering future implications",
            "indulgence": "Balancing enjoyment and restraint"
        }
        
        # Enhance context with feedback suggestions
        context = cultural_context[dimension]
        if suggestions:
            context = f"{context}, incorporating: {'; '.join(suggestions)}"
        
        # Add emphasis for priority dimensions
        if is_priority:
            context = f"[Priority] {context}"
            
        # Create base adapted prompt
        adapted_prompt = f"{context}, {guideline}: {original_prompt}"
        
        # Add character tone only if enabled
        if self.use_character_tone:
            tone = self._get_character_tone()
            adapted_prompt = f"Using a tone that is {tone}, respond to: {adapted_prompt}"
        
        return adapted_prompt

def create_cultural_prompt(dimensions: CulturalDimensions) -> str:
    """Create a prompt that guides response generation based on cultural dimensions."""
    guidelines = dimensions.get_dimension_guidelines()
    return "\n".join(guidelines.values())

# Example country dimensions (to be expanded)
COUNTRY_DIMENSIONS = {
    "japan": CulturalDimensions(
        power_distance=54,
        individualism=46,
        masculinity=95,
        uncertainty_avoidance=92,
        long_term_orientation=88,
        indulgence=42
    ),
    "usa": CulturalDimensions(
        power_distance=40,
        individualism=91,
        masculinity=62,
        uncertainty_avoidance=46,
        long_term_orientation=26,
        indulgence=68
    ),
    "india": CulturalDimensions(
        power_distance=70,
        individualism=30,
        masculinity=50,
        uncertainty_avoidance=80,
        long_term_orientation=40,
        indulgence=20
    ),
    "china": CulturalDimensions(
        power_distance=80,
        individualism=20,
        masculinity=50,
        uncertainty_avoidance=90,
        long_term_orientation=30,
        indulgence=10
    ),
    "brazil": CulturalDimensions(
        power_distance=60,
        individualism=40,
        masculinity=70,
        uncertainty_avoidance=50,
        long_term_orientation=60,
        indulgence=30
    ),
    "russia": CulturalDimensions(
        power_distance=80,
        individualism=20,
        masculinity=70,
        uncertainty_avoidance=90,
        long_term_orientation=40,
        indulgence=10
    ),
    "south_korea": CulturalDimensions(
        power_distance=70,
        individualism=30,
        masculinity=80,
        uncertainty_avoidance=90,
        long_term_orientation=40,
        indulgence=20
    ),
    "uae": CulturalDimensions(
        power_distance=70,
        individualism=80,
        masculinity=80,
        uncertainty_avoidance=90,
        long_term_orientation=40,
        indulgence=70
    ),
}
