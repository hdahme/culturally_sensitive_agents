"""
LangGraph orchestration for culturally sensitive prompt adaptation and evaluation.
"""
from typing import Dict, List, Any, Tuple, TypedDict, Annotated
from pathlib import Path
import json
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from operator import itemgetter

from hofstede import (
    CulturalDimensions,
    PromptAdaptation,
    CultureAwareAdapter,
    create_cultural_prompt,
    COUNTRY_DIMENSIONS
)

class WorkflowState(TypedDict):
    """State maintained through the workflow."""
    country: str
    character: Dict[str, Any]
    dimensions: CulturalDimensions
    original_prompt: str
    adapted_prompts: List[PromptAdaptation]
    current_adaptation: PromptAdaptation
    evaluation_results: List[Dict[str, Any]]
    comparison_metrics: Dict[str, Any]
    iteration: int
    feedback_history: List[Dict[str, Any]]

class CulturalOrchestrator:
    """Orchestrates prompt adaptation and cultural acceptance evaluation."""
    
    def __init__(self, character_file: Path, acceptance_threshold: float = 0.7, max_iterations: int = 3, use_character_tone: bool = False):
        self.character_file = character_file
        self.acceptance_threshold = acceptance_threshold
        self.max_iterations = max_iterations
        self.use_character_tone = use_character_tone
        self.llm = ChatOpenAI(model="gpt-4o")
        self.workflow = self._create_workflow()
        
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow for prompt adaptation and evaluation."""
        
        # Create the workflow graph
        workflow = StateGraph(WorkflowState)
        
        # Add nodes for each step
        workflow.add_node("initialize", self._initialize_state)
        workflow.add_node("adapt_prompts", self._adapt_prompts)
        workflow.add_node("evaluate_adaptations", self._evaluate_adaptations)
        workflow.add_node("calculate_comparisons", self._calculate_comparisons)
        workflow.add_node("check_acceptance", self._check_acceptance)
        workflow.add_node("generate_feedback", self._generate_feedback)
        
        # Define edges
        workflow.add_edge(START, "initialize")
        workflow.add_edge("initialize", "adapt_prompts")
        workflow.add_edge("adapt_prompts", "evaluate_adaptations")
        workflow.add_edge("evaluate_adaptations", "calculate_comparisons")
        workflow.add_edge("calculate_comparisons", "check_acceptance")
        
        # Add conditional edges based on acceptance check
        workflow.add_conditional_edges(
            "check_acceptance",
            self._should_continue_iteration,
            {
                True: "generate_feedback",
                False: END
            }
        )
        workflow.add_edge("generate_feedback", "adapt_prompts")
        
        # Compile the workflow
        return workflow.compile()
    
    def _initialize_state(self, state: WorkflowState) -> WorkflowState:
        """Initialize the workflow state with character and cultural information."""
        country = state["country"]
        if country not in COUNTRY_DIMENSIONS:
            raise ValueError(f"No cultural dimensions defined for {country}")
            
        dimensions = COUNTRY_DIMENSIONS[country]
        adapter = CultureAwareAdapter(
            self.character_file,
            dimensions,
            use_character_tone=self.use_character_tone
        )
        
        return {
            **state,
            "dimensions": dimensions,
            "character": adapter.character_data,
            "adapted_prompts": [],
            "evaluation_results": [],
            "comparison_metrics": {},
            "iteration": 0,
            "feedback_history": []
        }
    
    def _adapt_prompts(self, state: WorkflowState) -> WorkflowState:
        """Generate culturally adapted versions of the original prompt."""
        adapter = CultureAwareAdapter(
            self.character_file,
            state["dimensions"],
            use_character_tone=self.use_character_tone
        )
        
        # If we have feedback history, use it to improve adaptation
        if state["feedback_history"]:
            feedback = state["feedback_history"][-1]
            adapted_prompts = adapter.adapt_prompt_with_feedback(
                state["original_prompt"],
                feedback
            )
        else:
            adapted_prompts = adapter.adapt_prompt(state["original_prompt"])
        
        return {
            **state,
            "adapted_prompts": adapted_prompts,
            "iteration": state["iteration"] + 1
        }
    
    def _evaluate_adaptations(self, state: WorkflowState) -> WorkflowState:
        """Evaluate each adapted prompt for cultural appropriateness."""
        evaluation_results = []
        
        for adaptation in state["adapted_prompts"]:
            # Create cultural context for response generation
            dimensions = state["dimensions"].to_dict()
            cultural_context = "\n".join([
                f"- {dim.replace('_', ' ').title()}: {value}"
                for dim, value in dimensions.items()
            ])
            
            # Get character details if enabled
            character_context = ""
            if self.use_character_tone:
                adapter = CultureAwareAdapter(self.character_file, state["dimensions"])
                character = adapter.character_data
                
                # Extract style guidelines
                style_all = character.get('style', {}).get('all', [])
                style_chat = character.get('style', {}).get('chat', [])
                
                # Get topics and adjectives
                topics = character.get('topics', [])
                adjectives = character.get('adjectives', [])
                
                # Get bio and lore
                bio = character.get('bio', [])
                lore = character.get('lore', [])
                
                character_context = f"""
                You are embodying the following character:
                Name: {character.get('name', 'AI Assistant')}
                
                Core Style:
                {chr(10).join(f"- {style}" for style in style_all)}
                
                Chat Style:
                {chr(10).join(f"- {style}" for style in style_chat)}
                
                Key Traits:
                {chr(10).join(f"- {adj}" for adj in adjectives[:5])}
                
                Background:
                {chr(10).join(f"- {detail}" for detail in bio[:3])}
                {chr(10).join(f"- {detail}" for detail in lore[:2])}
                
                Expertise Topics:
                {', '.join(topics[:5])}
                
                Stay true to this character's unique style and personality while responding.
                Remember: {character.get('system', 'Never act like an assistant.')}
                """
                
                # Generate cultural adaptation suggestions for the character
                cultural_suggestions = self._generate_character_cultural_suggestions(
                    state["country"],
                    state["dimensions"],
                    character
                )
                
                # Add cultural suggestions to the context if available
                if cultural_suggestions:
                    character_context += f"\n\nCultural Adaptation Suggestions:\n{cultural_suggestions}"
            
            # Generate cultural suggestions (independent of character)
            cultural_suggestions = self._generate_cultural_suggestions(
                state["country"],
                state["dimensions"]
            )
            
            # First, generate responses for both original and adapted prompts
            original_response_prompt = f"""
            Respond to this prompt:
            {adaptation.original_prompt}
            
            Keep your response concise but informative.
            """
            
            adapted_response_prompt = f"""
            You are responding in a culturally appropriate way for {state['country'].upper()}.
            
            Cultural Context:
            {cultural_context}
            
            Cultural Guidelines:
            {cultural_suggestions}
            
            Target Dimension: {adaptation.target_dimension}
            Adaptation Focus: {adaptation.adaptation_explanation}
            
            Respond to this culturally-adapted prompt:
            {adaptation.adapted_prompt}
            
            Consider ALL cultural dimensions in your response.
            Keep your response concise but informative.
            """
            
            # Get responses
            original_response = self.llm.invoke([HumanMessage(content=original_response_prompt)])
            adapted_response = self.llm.invoke([HumanMessage(content=adapted_response_prompt)])
            
            # Store responses in the adaptation object
            adaptation.original_response = original_response.content
            adaptation.adapted_response = adapted_response.content
            
            # First evaluate the original response
            original_eval_prompt = f"""
            Evaluate the cultural appropriateness{' and character consistency' if self.use_character_tone else ''} of this response for {state['country']}.
            
            {character_context if self.use_character_tone else ''}
            
            Cultural Context:
            {cultural_context}
            
            Cultural Guidelines:
            {cultural_suggestions}
            
            Original Prompt: {adaptation.original_prompt}
            Response: {adaptation.original_response}
            
            Consider:
            1. Cultural authenticity and appropriateness of the response
            2. {'How well the characters personality and tone are maintained' if self.use_character_tone else 'How well cultural norms are respected'}
            3. Potential cultural misunderstandings
            4. How well the response incorporates cultural dimensions
            5. {'Whether the characters voice enhances or detracts from cultural appropriateness' if self.use_character_tone else 'Whether the response shows cultural sensitivity'}
            
            Return your evaluation as a valid JSON object with these exact fields:
            {{
                "cultural_acceptance_score": <float between 0-1>,
                "character_consistency_score": <float between 0-1>,
                "explanation": <string>,
                "cultural_suggestions": <string>,
                "character_analysis": {{
                    "tone_preservation": <float between 0-1>,
                    "personality_match": <float between 0-1>,
                    "voice_authenticity": <float between 0-1>,
                    "comments": <string>
                }},
                "dimension_analysis": {{
                    <dimension_name>: {{
                        "score": <float between 0-1>,
                        "comments": <string>
                    }}
                }}
            }}
            """
            
            # Get the original response evaluation
            original_llm_response = self.llm.invoke([HumanMessage(content=original_eval_prompt)])
            original_llm_response = original_llm_response.content.strip("```json\n").strip("\n```")
            try:
                original_evaluation = json.loads(original_llm_response)
                adaptation.original_cultural_score = original_evaluation["cultural_acceptance_score"]
                if self.use_character_tone:
                    adaptation.original_consistency_score = original_evaluation["character_consistency_score"]
            except json.JSONDecodeError:
                adaptation.original_cultural_score = 0.0
                adaptation.original_consistency_score = 0.0
                original_evaluation = {
                    "cultural_acceptance_score": 0.0,
                    "character_consistency_score": 0.0,
                    "explanation": "Failed to parse evaluation",
                    "cultural_suggestions": "",
                    "character_analysis": {
                        "tone_preservation": 0.0,
                        "personality_match": 0.0,
                        "voice_authenticity": 0.0,
                        "comments": "Failed to parse evaluation"
                    },
                    "dimension_analysis": {}
                }
            
            # Now evaluate the adapted response
            adapted_eval_prompt = f"""
            Evaluate the cultural appropriateness{' and character consistency' if self.use_character_tone else ''} of the following responses for {state['country']}.
            
            {character_context if self.use_character_tone else ''}
            
            Cultural Context:
            {cultural_context}
            
            Cultural Guidelines:
            {cultural_suggestions}
            
            Consider previous feedback if available: {state['feedback_history']}
            
            Original Prompt: {adaptation.original_prompt}
            Original Response: {adaptation.original_response}
            Original Cultural Score: {adaptation.original_cultural_score:.2f}
            {'Original Character Score: ' + str(adaptation.original_consistency_score) if self.use_character_tone else ''}
            
            Adapted Prompt: {adaptation.adapted_prompt}
            Adapted Response: {adaptation.adapted_response}
            
            Target Dimension: {adaptation.target_dimension}
            Adaptation Explanation: {adaptation.adaptation_explanation}
            
            Consider:
            1. Cultural authenticity and appropriateness compared to the original response
            2. {'How well the characters personality and tone are maintained' if self.use_character_tone else 'How well cultural norms are respected'}
            3. {'Balance between cultural adaptation and character authenticity' if self.use_character_tone else 'Effectiveness of cultural adaptation'}
            4. Potential cultural misunderstandings or improvements
            5. How well the response incorporates all cultural dimensions
            6. {'Whether the characters voice enhances or detracts from cultural appropriateness' if self.use_character_tone else 'Whether the response shows cultural sensitivity'}
            
            Return your evaluation as a valid JSON object with these exact fields:
            {{
                "cultural_acceptance_score": <float between 0-1>,
                "character_consistency_score": <float between 0-1>,
                "explanation": <string>,
                "cultural_suggestions": <string>,
                "strengths": <list of strings>,
                "weaknesses": <list of strings>,
                "improvement_suggestions": <list of strings>,
                "character_analysis": {{
                    "tone_preservation": <float between 0-1>,
                    "personality_match": <float between 0-1>,
                    "voice_authenticity": <float between 0-1>,
                    "comments": <string>
                }},
                "dimension_analysis": {{
                    <dimension_name>: {{
                        "score": <float between 0-1>,
                        "comments": <string>
                    }}
                }}
            }}
            """
            
            # Get the adapted response evaluation
            adapted_llm_response = self.llm.invoke([HumanMessage(content=adapted_eval_prompt)])
            adapted_llm_response = adapted_llm_response.content.strip("```json\n").strip("\n```")
            try:
                adapted_evaluation = json.loads(adapted_llm_response)
                adaptation.cultural_acceptance_score = adapted_evaluation["cultural_acceptance_score"]
                adaptation.character_consistency_score = adapted_evaluation["character_consistency_score"]
            except json.JSONDecodeError:
                adaptation.cultural_acceptance_score = 0.0
                adaptation.character_consistency_score = 0.0
                adapted_evaluation = {
                    "cultural_acceptance_score": 0.0,
                    "character_consistency_score": 0.0,
                    "explanation": "Failed to parse evaluation",
                    "cultural_suggestions": "",
                    "strengths": [],
                    "weaknesses": ["JSON parsing failed"],
                    "improvement_suggestions": ["Retry evaluation"],
                    "character_analysis": {
                        "tone_preservation": 0.0,
                        "personality_match": 0.0,
                        "voice_authenticity": 0.0,
                        "comments": ""
                    },
                    "dimension_analysis": {}
                }
            
            # Store scores and evaluation results in the adaptation object
            adaptation.evaluation_results.append(adapted_evaluation)  # Store the full evaluation
            evaluation_results.append(adapted_evaluation)
            
        return {**state, "evaluation_results": evaluation_results}
    
    def _calculate_comparisons(self, state: WorkflowState) -> WorkflowState:
        """Calculate comparison metrics between original and adapted prompts."""
        metrics = {
            "average_improvement": {
                "cultural_acceptance": 0.0,
                "character_consistency": 0.0
            },
            "dimension_improvements": {},
            "total_adaptations": len(state["adapted_prompts"]),
            "iteration": state["iteration"]
        }
        
        # Calculate average improvements
        for adaptation in state["adapted_prompts"]:
            metrics["average_improvement"]["cultural_acceptance"] += adaptation.cultural_acceptance_score
            metrics["average_improvement"]["character_consistency"] += adaptation.character_consistency_score
            
            # Track improvements by dimension
            if adaptation.target_dimension not in metrics["dimension_improvements"]:
                metrics["dimension_improvements"][adaptation.target_dimension] = {
                    "cultural_acceptance": adaptation.cultural_acceptance_score,
                    "character_consistency": adaptation.character_consistency_score
                }
        
        # Calculate averages
        total = len(state["adapted_prompts"])
        metrics["average_improvement"]["cultural_acceptance"] /= total
        metrics["average_improvement"]["character_consistency"] /= total
            
        return {**state, "comparison_metrics": metrics}
    
    def _check_acceptance(self, state: WorkflowState) -> WorkflowState:
        """Check if the cultural acceptance scores meet our threshold."""
        return state
    
    def _should_continue_iteration(self, state: WorkflowState) -> bool:
        """Determine if we should continue iterating based on scores and iteration count."""
        avg_acceptance = state["comparison_metrics"]["average_improvement"]["cultural_acceptance"]
        return (
            avg_acceptance < self.acceptance_threshold
            and state["iteration"] < self.max_iterations
        )
    
    def _generate_feedback(self, state: WorkflowState) -> WorkflowState:
        """Generate feedback for improving cultural adaptation."""
        prompt = f"""
        Based on the evaluation results, generate specific feedback for improving
        cultural adaptation for {state['country']}.
        
        Current Results:
        {json.dumps(state['evaluation_results'], indent=2)}
        
        Previous Feedback:
        {json.dumps(state['feedback_history'], indent=2)}
        
        Return your feedback as a valid JSON object with these exact fields:
        {{
            "dimension_feedback": {{
                <dimension>: {{
                    "issues": <list of strings>,
                    "suggestions": <list of strings>
                }}
            }},
            "general_feedback": <list of strings>,
            "priority_dimensions": <list of strings>
        }}
        """
        
        llm_response = self.llm.invoke([HumanMessage(content=prompt)])
        llm_response = llm_response.content.strip("```json\n").strip("\n```")
        try:
            feedback = json.loads(llm_response)
        except json.JSONDecodeError:
            feedback = {
                "dimension_feedback": {},
                "general_feedback": ["Failed to generate specific feedback"],
                "priority_dimensions": []
            }
        
        return {
            **state,
            "feedback_history": [*state["feedback_history"], feedback]
        }
    
    def _generate_character_cultural_suggestions(
        self,
        country: str,
        dimensions: CulturalDimensions,
        character: Dict[str, Any]
    ) -> str:
        """Generate cultural adaptation suggestions for the character."""
        prompt = f"""
        Analyze this character definition and suggest cultural adaptations for {country}.
        
        Character Definition:
        {json.dumps(character, indent=2)}
        
        Cultural Dimensions:
        {json.dumps(dimensions.to_dict(), indent=2)}
        
        Generate cultural adaptation suggestions as a YAML structure with these sections:
        1. cultural_style: List of style guidelines for this culture
        2. cultural_topics: Additional topics relevant to this culture
        3. cultural_adjectives: Culture-specific character traits
        4. cultural_examples: Example responses in this cultural context
        5. cultural_taboos: Things to avoid in this culture
        
        Return only the YAML structure without any other text.
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            logger.error(f"Failed to generate cultural suggestions: {str(e)}")
            return ""
    
    def _generate_cultural_suggestions(
        self,
        country: str,
        dimensions: CulturalDimensions
    ) -> str:
        """Generate cultural adaptation suggestions independent of character."""
        prompt = f"""
        Generate cultural adaptation suggestions for {country}.
        
        Cultural Dimensions:
        {json.dumps(dimensions.to_dict(), indent=2)}
        
        Generate cultural adaptation suggestions as a YAML structure with these sections:
        1. cultural_style: List of communication style guidelines for this culture
        2. cultural_topics: Topics that are important or sensitive in this culture
        3. cultural_values: Key cultural values and beliefs
        4. cultural_examples: Example phrasings and approaches in this cultural context
        5. cultural_taboos: Things to avoid in this culture
        6. cultural_etiquette: Important etiquette and protocol considerations
        
        Return only the YAML structure without any other text.
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            logger.error(f"Failed to generate cultural suggestions: {str(e)}")
            return ""
    
    def adapt_and_evaluate(self, country: str, prompt: str) -> Tuple[List[PromptAdaptation], Dict[str, Any]]:
        """Run the workflow to adapt and evaluate a prompt for a specific culture."""
        initial_state = {
            "country": country,
            "original_prompt": prompt,
            "feedback_history": []
        }
        final_state = self.workflow.invoke(initial_state)
        
        return final_state["adapted_prompts"], final_state["comparison_metrics"] 