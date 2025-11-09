"""
AI Insights Service - Generates business insights using LangChain and OpenAI
"""

import os
import logging
from typing import Dict, List, Optional, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

from src.api.schemas.responses import AIInsights

logger = logging.getLogger(__name__)
load_dotenv()

class AIInsightsService:
    """Service for generating AI-powered business insights using LangChain"""

    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.7, max_tokens: int = 1000):
        """
        Initialize AI Insights Service with LangChain

        Args:
            model_name: OpenAI model to use (default: gpt-4)
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens in response
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize ChatOpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        logger.info(f"Api key: {'found' if api_key else 'not found'}")
        if not api_key:
            logger.warning("OPENAI_API_KEY not found. AI insights will be unavailable.")
            self.llm = None
        else:
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=api_key
            )

        # Initialize output parser for structured responses
        self.parser = PydanticOutputParser(pydantic_object=AIInsights)

        # Create prompt templates
        self._init_prompts()

    def _init_prompts(self):
        """Initialize LangChain prompt templates"""

        # Format instructions for the parser
        format_instructions = self.parser.get_format_instructions()

        # Forward prediction prompt template
        self.forward_prompt = PromptTemplate(
            template="""You are a business intelligence analyst specializing in fitness studio revenue predictions.

Analyze the following revenue prediction data and provide actionable business insights.

PREDICTION DATA:
- Studio ID: {studio_id}
- Total Projected Revenue: ${total_revenue:,.2f}
- Average Confidence: {avg_confidence:.1%}
- Number of Months: {num_months}
- Monthly Breakdown: {monthly_predictions}

TECHNICAL EXPLANATION (SHAP values):
{explanation}

QUICK WIN RECOMMENDATIONS:
{quick_wins}

PRODUCT/SERVICE ANALYSIS DATA:
{product_recommendations}

IMPORTANT: Based on the product correlation data above, create engaging, natural language recommendations with emojis.
DO NOT return product recommendations in JSON format. Instead, weave them into your narrative recommendations section 
with clear, actionable advice. Use emojis like 🎯 for promotions, 💎 for premium offerings, 📦 for packages, 
🛍️ for retail, 💪 for services, ⚠️ for items to review.

Transform the correlation data into compelling, human-readable insights that a studio owner can immediately understand 
and act upon. Focus on WHY each product should be promoted or reconsidered based on its performance metrics.

PRESENTATION STYLE:
Make your insights engaging and intuitive by:
- Using relevant emojis naturally throughout the text (📊 for data/charts, 💰 for revenue, 👥 for members, 🔄 for retention, 📈 for growth, 📉 for decline, ⚡ for quick wins, 🎯 for targets, ✨ for opportunities, 🌟 for highlights)
- Embedding visual descriptions (e.g., "📊 Revenue is trending upward..." or "💰 Strong revenue growth driven by..." or "⚡ Quick win opportunity identified...")
- Using clear visual markers to highlight key points and make insights scannable
- Keeping technical concepts accessible with intuitive, visual language
- Making numerical data come alive with contextual emojis

{format_instructions}
""",
            input_variables=["studio_id", "total_revenue", "avg_confidence", "num_months",
                           "monthly_predictions", "explanation", "quick_wins", "product_recommendations"],
            partial_variables={"format_instructions": format_instructions}
        )

        # Inverse prediction prompt template
        self.inverse_prompt = PromptTemplate(
            template="""You are a business strategy consultant specializing in fitness studio optimization.

Analyze the following optimization recommendation and provide actionable strategic insights.

OPTIMIZATION DATA:
- Studio ID: {studio_id}
- Target Revenue: ${target_revenue:,.2f}
- Achievable Revenue: ${achievable_revenue:,.2f}
- Achievement Rate: {achievement_rate:.1%}
- Confidence Score: {confidence_score:.1%}

RECOMMENDED LEVER CHANGES:
{lever_changes}

ACTION PLAN:
{action_plan}

SENSITIVITY ANALYSIS:
{sensitivity}

FEASIBILITY ASSESSMENT:
{feasibility}

Provide strategic insights that help the studio owner understand:
1. What the optimization means in practical terms
2. Which actions to prioritize and why
3. Potential challenges and how to mitigate them
4. Why the confidence level is what it is

PRESENTATION STYLE:
Make your insights engaging and intuitive by:
- Using relevant emojis naturally throughout the text (🎯 for targets/goals, 🚀 for actions/growth, 📊 for metrics/data, ⚠️ for challenges/warnings, ✅ for achievements/completions, 🔧 for levers/adjustments, 💡 for strategies/ideas, 📈 for improvements, ⚖️ for balance/trade-offs)
- Embedding visual descriptions (e.g., "🎯 Target revenue goal..." or "🚀 Action plan to boost performance..." or "⚠️ Potential challenge to consider...")
- Using clear visual markers to make action items and priorities scannable
- Presenting optimization recommendations with visual cues that guide decision-making
- Making strategic concepts accessible with intuitive, visual language

{format_instructions}
""",
            input_variables=["studio_id", "target_revenue", "achievable_revenue", "achievement_rate",
                           "confidence_score", "lever_changes", "action_plan", "sensitivity", "feasibility"],
            partial_variables={"format_instructions": format_instructions}
        )

        # Partial prediction prompt template
        self.partial_prompt = PromptTemplate(
            template="""You are a business analyst helping fitness studios understand their operational levers.

Analyze the following lever predictions and provide actionable insights.

PARTIAL PREDICTION DATA:
- Studio ID: {studio_id}
- Input Levers: {input_levers}
- Predicted Levers: {predicted_levers}
- Overall Confidence: {confidence:.1%}

TECHNICAL EXPLANATION (SHAP):
Model Context (Revenue/Members/Retention predictions):
{model_context}

Lever-Specific Insights (How input levers influence output levers):
{lever_insights}

PRODUCT/SERVICE ANALYSIS DATA (Targeted by Output Levers):
{product_recommendations}

NOTES:
{notes}

IMPORTANT: Transform the product analysis data above into engaging, natural language recommendations with emojis.
DO NOT return product recommendations in JSON format. Instead, integrate them naturally into your recommendations 
section with clear, actionable advice. Use emojis like 🎯 for promotions, 💎 for premium offerings, 📦 for packages, 
🛍️ for retail, 💪 for services, ⚠️ for items to review.

Based on the SHAP analysis and targeted product correlations, explain:
1. What the model predicts about the studio's trajectory (revenue, members, retention)
2. How the input levers you provided influence the output levers being predicted
3. Which input levers have the strongest impact on each output lever
4. Which products/services to promote (with emojis and natural language) based on the specific output levers being optimized
5. How product recommendations align with predicted lever targets
6. Specific actions to optimize these lever relationships and product mix

PRESENTATION STYLE:
Make your insights engaging and intuitive by:
- Using relevant emojis naturally throughout the text (⚙️ for input levers, 🎯 for output levers, 🔗 for relationships/connections, 💡 for insights/findings, 📊 for predictions/data, 🔄 for influence/impact, ⬆️ for increases, ⬇️ for decreases, 🌟 for key drivers, 🎨 for product/service mix)
- Embedding visual descriptions (e.g., "⚙️ Input lever adjustments..." or "🔗 Strong connection between..." or "🎯 Output lever targeting...")
- Using clear visual markers to show input→output relationships
- Making SHAP explanations intuitive with visual language that shows cause and effect
- Presenting lever interactions with visual cues that clarify dependencies

{format_instructions}
""",
            input_variables=["studio_id", "input_levers", "predicted_levers", "confidence", 
                           "model_context", "lever_insights", "product_recommendations", "notes"],
            partial_variables={"format_instructions": format_instructions}
        )

        # Scenario comparison prompt template
        self.scenario_prompt = PromptTemplate(
            template="""You are a business strategy consultant helping fitness studios compare optimization scenarios.

Analyze the following scenario comparison and provide decision-making insights.

SCENARIO COMPARISON:
{scenarios}

COMPARISON METRICS:
- Number of scenarios: {num_scenarios}
- Revenue range: ${min_revenue:,.2f} - ${max_revenue:,.2f}
- Average achievement rate: {avg_achievement:.1%}

Provide insights to help decision-makers:
1. Which scenario is recommended and why
2. Trade-offs between scenarios
3. Risk assessment for each scenario
4. Implementation considerations

PRESENTATION STYLE:
Make your insights engaging and intuitive by:
- Using relevant emojis naturally throughout the text (⚖️ for trade-offs/comparisons, 🏆 for recommended/best options, ⚠️ for risks/concerns, 📊 for metrics/comparisons, ✅ for pros/advantages, ❌ for cons/disadvantages, 💼 for business decisions, 🎯 for goals/targets, 🌟 for standout features, 📈 for performance indicators)
- Embedding visual descriptions (e.g., "🏆 Recommended scenario..." or "⚖️ Trade-off between..." or "⚠️ Risk consideration...")
- Using clear visual markers to differentiate between scenarios
- Making scenario differences immediately apparent with visual cues
- Presenting decision-making insights with visual guides that support strategic choices

{format_instructions}
""",
            input_variables=["scenarios", "num_scenarios", "min_revenue", "max_revenue", "avg_achievement"],
            partial_variables={"format_instructions": format_instructions}
        )

    def generate_forward_insights(
        self,
        studio_id: str,
        total_revenue: float,
        avg_confidence: float,
        predictions: List[Dict],
        explanation: Optional[Dict] = None,
        quick_wins: Optional[List[Dict]] = None,
        product_recommendations: Optional[Dict] = None
    ) -> Optional[AIInsights]:
        """
        Generate AI insights for forward predictions using LangChain LCEL

        Args:
            studio_id: Studio identifier
            total_revenue: Total projected revenue
            avg_confidence: Average confidence score
            predictions: List of monthly predictions
            explanation: SHAP explanation data
            quick_wins: Quick win recommendations
            product_recommendations: Product/service recommendations from analyzer

        Returns:
            AIInsights object or None if generation fails
        """
        if not self.llm:
            logger.warning("LLM not initialized. Skipping AI insights generation.")
            return None

        try:
            # Format monthly predictions
            monthly_str = "\n".join([
                f"  Month {p.get('month', 'N/A')}: ${p.get('revenue', 0):,.2f} "
                f"(Members: {p.get('member_count', 0)}, Retention: {p.get('retention_rate', 0):.1%})"
                for p in predictions
            ])

            # Format explanation
            exp_str = self._format_explanation(explanation) if explanation else "No detailed explanation available"

            # Format quick wins
            qw_str = self._format_quick_wins(quick_wins) if quick_wins else "No quick wins identified"
            
            # Format product recommendations
            prod_str = self._format_product_recommendations(product_recommendations) if product_recommendations else "No product analysis available"

            # Create LCEL chain: prompt | model | parser
            chain = self.forward_prompt | self.llm | self.parser

            # Invoke chain with inputs
            insights = chain.invoke({
                "studio_id": studio_id,
                "total_revenue": total_revenue,
                "avg_confidence": avg_confidence,
                "num_months": len(predictions),
                "monthly_predictions": monthly_str,
                "explanation": exp_str,
                "quick_wins": qw_str,
                "product_recommendations": prod_str
            })

            return insights

        except Exception as e:
            logger.error(f"Error generating forward insights: {str(e)}")
            return None

    def generate_inverse_insights(
        self,
        studio_id: str,
        target_revenue: float,
        achievable_revenue: float,
        achievement_rate: float,
        confidence_score: float,
        lever_changes: List[Dict],
        action_plan: List[Dict],
        sensitivity: Optional[Dict] = None,
        feasibility: Optional[Dict] = None
    ) -> Optional[AIInsights]:
        """
        Generate AI insights for inverse predictions using LangChain LCEL

        Args:
            studio_id: Studio identifier
            target_revenue: Target revenue goal
            achievable_revenue: Achievable revenue with constraints
            achievement_rate: Percentage of target achieved
            confidence_score: Optimization confidence
            lever_changes: Recommended lever changes
            action_plan: Prioritized action items
            sensitivity: Sensitivity analysis data
            feasibility: Feasibility assessment data

        Returns:
            AIInsights object or None if generation fails
        """
        if not self.llm:
            logger.warning("LLM not initialized. Skipping AI insights generation.")
            return None

        try:
            # Format lever changes
            lever_str = self._format_lever_changes(lever_changes)

            # Format action plan
            action_str = self._format_action_plan(action_plan)

            # Format sensitivity
            sens_str = self._format_sensitivity(sensitivity) if sensitivity else "No sensitivity data available"

            # Format feasibility
            feas_str = self._format_feasibility(feasibility) if feasibility else "No feasibility data available"

            # Create LCEL chain
            chain = self.inverse_prompt | self.llm | self.parser

            # Invoke chain
            insights = chain.invoke({
                "studio_id": studio_id,
                "target_revenue": target_revenue,
                "achievable_revenue": achievable_revenue,
                "achievement_rate": achievement_rate,
                "confidence_score": confidence_score,
                "lever_changes": lever_str,
                "action_plan": action_str,
                "sensitivity": sens_str,
                "feasibility": feas_str
            })

            return insights

        except Exception as e:
            logger.error(f"Error generating inverse insights: {str(e)}")
            return None

    def generate_partial_insights(
        self,
        studio_id: str,
        input_levers: Dict[str, float],
        predicted_levers: List[Dict],
        confidence: float,
        notes: Optional[str] = None,
        explanation: Optional[Dict] = None,
        product_recommendations: Optional[Dict] = None
    ) -> Optional[AIInsights]:
        """
        Generate AI insights for partial predictions using LangChain LCEL

        Args:
            studio_id: Studio identifier
            input_levers: Levers provided as input
            predicted_levers: Predicted lever values
            confidence: Overall confidence score
            notes: Additional notes or warnings
            explanation: SHAP explanation with model context and lever insights
            product_recommendations: Targeted product recommendations based on output levers

        Returns:
            AIInsights object or None if generation fails
        """
        if not self.llm:
            logger.warning("LLM not initialized. Skipping AI insights generation.")
            return None

        try:
            # Format input levers
            input_str = "\n".join([f"  {k}: {v}" for k, v in input_levers.items()])

            # Format predicted levers
            pred_str = self._format_predicted_levers(predicted_levers)

            # Format model context from SHAP explanation
            model_context_str = self._format_model_context(explanation) if explanation else "No model context available"
            
            # Format lever insights from SHAP explanation
            lever_insights_str = self._format_lever_insights(explanation) if explanation else "No lever insights available"

            # Format product recommendations for targeted output levers
            prod_str = self._format_product_recommendations(product_recommendations) if product_recommendations else "No product analysis available"

            # Create LCEL chain
            chain = self.partial_prompt | self.llm | self.parser

            # Invoke chain
            insights = chain.invoke({
                "studio_id": studio_id,
                "input_levers": input_str,
                "predicted_levers": pred_str,
                "confidence": confidence,
                "model_context": model_context_str,
                "lever_insights": lever_insights_str,
                "product_recommendations": prod_str,
                "notes": notes or "No additional notes"
            })

            return insights

        except Exception as e:
            logger.error(f"Error generating partial insights: {str(e)}")
            return None

    def generate_scenario_comparison_insights(
        self,
        scenarios: List[Dict],
        comparison_metrics: Dict[str, Any]
    ) -> Optional[AIInsights]:
        """
        Generate AI insights for scenario comparison using LangChain LCEL

        Args:
            scenarios: List of scenario data
            comparison_metrics: Aggregated comparison metrics

        Returns:
            AIInsights object or None if generation fails
        """
        if not self.llm:
            logger.warning("LLM not initialized. Skipping AI insights generation.")
            return None

        try:
            # Format scenarios
            scenario_str = self._format_scenarios(scenarios)

            # Create LCEL chain
            chain = self.scenario_prompt | self.llm | self.parser

            # Invoke chain
            insights = chain.invoke({
                "scenarios": scenario_str,
                "num_scenarios": len(scenarios),
                "min_revenue": comparison_metrics.get("min_revenue", 0),
                "max_revenue": comparison_metrics.get("max_revenue", 0),
                "avg_achievement": comparison_metrics.get("avg_achievement", 0)
            })

            return insights

        except Exception as e:
            logger.error(f"Error generating scenario comparison insights: {str(e)}")
            return None

    # Helper methods for formatting data

    def _format_explanation(self, explanation: Dict) -> str:
        """Format SHAP explanation into readable text"""
        if not explanation:
            return "No explanation available"

        top_drivers = explanation.get("top_drivers", [])
        if not top_drivers:
            return "No key drivers identified"

        lines = ["Key drivers (SHAP analysis):"]
        for i, driver in enumerate(top_drivers[:5], 1):
            feature = driver.get("feature", "Unknown")
            contribution = driver.get("contribution", 0)
            lines.append(f"  {i}. {feature}: ${contribution:,.2f} impact")

        return "\n".join(lines)

    def _format_quick_wins(self, quick_wins: List[Dict]) -> str:
        """Format quick wins into readable text"""
        if not quick_wins:
            return "No quick wins available"

        lines = ["Quick win opportunities:"]
        for i, qw in enumerate(quick_wins[:5], 1):
            lever = qw.get("lever", "Unknown")
            impact = qw.get("expected_impact", 0)
            change = qw.get("change_required", "N/A")
            lines.append(f"  {i}. Adjust {lever} by {change} → ${impact:,.2f} impact")

        return "\n".join(lines)

    def _format_lever_changes(self, lever_changes: List[Dict]) -> str:
        """Format lever changes into readable text"""
        if not lever_changes:
            return "No lever changes"

        lines = ["Recommended changes:"]
        for lc in lever_changes:
            name = lc.get("lever_name", "Unknown")
            current = lc.get("current_value", 0)
            recommended = lc.get("recommended_value", 0)
            change_pct = lc.get("change_percentage", 0)
            priority = lc.get("priority", "N/A")
            lines.append(f"  - {name}: {current} → {recommended} ({change_pct:+.1f}%) [Priority: {priority}]")

        return "\n".join(lines)

    def _format_action_plan(self, action_plan: List[Dict]) -> str:
        """Format action plan into readable text"""
        if not action_plan:
            return "No action plan available"

        lines = ["Action items:"]
        for action in action_plan:
            priority = action.get("priority", "N/A")
            lever = action.get("lever", "Unknown")
            action_desc = action.get("action", "N/A")
            impact = action.get("expected_impact", 0)
            timeline = action.get("timeline_weeks", 0)
            lines.append(f"  {priority}. {action_desc} ({lever})")
            lines.append(f"     Impact: ${impact:,.2f} | Timeline: {timeline} weeks")

        return "\n".join(lines)

    def _format_sensitivity(self, sensitivity: Dict) -> str:
        """Format sensitivity analysis into readable text"""
        if not sensitivity:
            return "No sensitivity analysis"

        lines = ["Lever sensitivity:"]
        for lever, score in sensitivity.items():
            lines.append(f"  - {lever}: {score:.2f}")

        return "\n".join(lines)

    def _format_feasibility(self, feasibility: Dict) -> str:
        """Format feasibility assessment into readable text"""
        if not feasibility:
            return "No feasibility assessment"

        overall = feasibility.get("overall_score", "N/A")
        difficulty = feasibility.get("difficulty", "N/A")
        timeline = feasibility.get("timeline", "N/A")

        return f"Overall feasibility: {overall}\nDifficulty: {difficulty}\nTimeline: {timeline}"

    def _format_predicted_levers(self, predicted_levers: List[Dict]) -> str:
        """Format predicted levers into readable text"""
        if not predicted_levers:
            return "No predicted levers"

        lines = ["Predicted values:"]
        for pl in predicted_levers:
            name = pl.get("lever_name", "Unknown")
            value = pl.get("predicted_value", 0)
            confidence = pl.get("confidence_score", 0)
            lines.append(f"  - {name}: {value} (confidence: {confidence:.1%})")

        return "\n".join(lines)

    def _format_scenarios(self, scenarios: List[Dict]) -> str:
        """Format scenarios for comparison"""
        if not scenarios:
            return "No scenarios available"

        lines = []
        for i, scenario in enumerate(scenarios, 1):
            lines.append(f"\nScenario {i}:")
            lines.append(f"  Target: ${scenario.get('target_revenue', 0):,.2f}")
            lines.append(f"  Achievable: ${scenario.get('achievable_revenue', 0):,.2f}")
            lines.append(f"  Achievement: {scenario.get('achievement_rate', 0):.1%}")
            lines.append(f"  Confidence: {scenario.get('confidence_score', 0):.1%}")

        return "\n".join(lines)
    
    def _format_product_recommendations(self, product_recs: Dict) -> str:
        """
        Format product/service recommendations into natural language for AI to summarize with emojis.
        Provides data summary that AI will transform into engaging narrative.
        """
        if not product_recs:
            return "No product recommendations available"
        
        lines = []
        
        # Products to promote
        promote = product_recs.get('promote', [])
        if promote:
            lines.append("HIGH-PERFORMING PRODUCTS (Strong revenue correlation - recommend promoting):")
            for i, prod in enumerate(promote[:5], 1):
                product = prod.get('product', 'Unknown')
                category = prod.get('category', 'Unknown')
                correlation = prod.get('correlation', 0)
                avg_revenue = prod.get('avg_revenue', 0)
                impact_score = prod.get('impact_score', 0)
                action_items = prod.get('action_items', [])
                
                # Create summary for this product
                lines.append(f"\n{product} ({category}):")
                lines.append(f"  - Correlation strength: {correlation:.1%} with key metrics")
                lines.append(f"  - Average revenue: ${avg_revenue:,.0f}/month")
                lines.append(f"  - Impact score: {impact_score:.1f}/100")
                
                if action_items:
                    lines.append(f"  - Suggested actions:")
                    for action in action_items[:3]:
                        lines.append(f"    • {action}")
        
        # Products to demote/review
        demote = product_recs.get('demote', [])
        if demote:
            lines.append("\n\nUNDERPERFORMING PRODUCTS (Weak correlation - consider reviewing):")
            for i, prod in enumerate(demote[:3], 1):
                product = prod.get('product', 'Unknown')
                category = prod.get('category', 'Unknown')
                correlation = prod.get('correlation', 0)
                reasoning = prod.get('reasoning', 'N/A')
                
                lines.append(f"\n{product} ({category}):")
                lines.append(f"  - Correlation: {correlation:.1%}")
                lines.append(f"  - Analysis: {reasoning}")
        
        if not lines:
            return "No significant product performance patterns identified in the data"
        
        lines.append("\n\nNOTE: Transform this data into engaging, actionable recommendations with emojis. Don't list as JSON.")
        return "\n".join(lines)

    def _format_model_context(self, explanation: Dict) -> str:
        """
        Format model context (revenue, members, retention predictions) from SHAP explanation
        
        Args:
            explanation: SHAP explanation dictionary with targets
            
        Returns:
            Formatted string with model predictions and key drivers
        """
        if not explanation or 'targets' not in explanation:
            return "No model context available"
        
        lines = []
        targets = explanation.get('targets', {})
        
        # Format revenue predictions
        revenue_targets = ['revenue_month_1', 'revenue_month_2', 'revenue_month_3']
        lines.append("Revenue Predictions:")
        for target in revenue_targets:
            if target in targets:
                target_exp = targets[target]
                prediction = target_exp.get('prediction', 0)
                month = target.split('_')[-1]
                lines.append(f"  Month {month}: ${prediction:,.2f}")
        
        # Format member prediction
        if 'member_count_month_3' in targets:
            member_exp = targets['member_count_month_3']
            prediction = member_exp.get('prediction', 0)
            lines.append(f"\nMember Count (Month 3): {prediction:.0f}")
        
        # Format retention prediction
        if 'retention_rate_month_3' in targets:
            retention_exp = targets['retention_rate_month_3']
            prediction = retention_exp.get('prediction', 0)
            lines.append(f"Retention Rate (Month 3): {prediction:.1%}")
        
        # Add overall summary from explanation
        if 'summary' in explanation:
            summary = explanation['summary']
            most_important = summary.get('most_important_features_overall', [])
            if most_important:
                lines.append("\nMost Important Features Overall:")
                for feat in most_important[:5]:
                    feature = feat.get('feature', 'Unknown')
                    importance = feat.get('total_importance', 0)
                    lines.append(f"  - {feature}: {importance:.2f}")
        
        return "\n".join(lines)

    def _format_lever_insights(self, explanation: Dict) -> str:
        """
        Format lever-specific insights showing input→output lever relationships
        
        Args:
            explanation: SHAP explanation dictionary with lever_insights
            
        Returns:
            Formatted string with lever relationships
        """
        if not explanation or 'lever_insights' not in explanation:
            return "No lever insights available"
        
        lever_insights = explanation.get('lever_insights', {})
        
        if not lever_insights:
            return "No lever insights generated"
        
        lines = []
        
        for output_lever, insights in lever_insights.items():
            lines.append(f"\n{output_lever.upper()}:")
            
            predicted_value = insights.get('predicted_value')
            if predicted_value is not None:
                lines.append(f"  Predicted Value: {predicted_value:.2f}")
            
            # Format key drivers
            key_drivers = insights.get('key_drivers', [])
            if key_drivers:
                lines.append("  Key Influencing Input Levers:")
                for driver in key_drivers:
                    input_lever = driver.get('input_lever', 'Unknown')
                    contribution = driver.get('contribution', 0)
                    via_target = driver.get('via_target', 'Unknown')
                    
                    direction = "↑ increases" if contribution > 0 else "↓ decreases"
                    lines.append(f"    - {input_lever} {direction} (contribution: {contribution:.2f} via {via_target})")
            
            # Add explanation text
            explanation_text = insights.get('explanation', '')
            if explanation_text:
                lines.append(f"  Explanation: {explanation_text}")
        
        return "\n".join(lines)
