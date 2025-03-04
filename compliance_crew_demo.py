"""
Compliance Crew Demo

This script demonstrates how to use the ComplianceCrewConnector to register
agents and crews in the graph database and execute compliance-related tasks.
"""

import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from AgentKG.src.compliance_crew_connector import ComplianceCrewConnector

# Load environment variables
load_dotenv()


def setup_compliance_crews():
    """Set up compliance crews and register them in the graph database"""
    try:
        # Initialize language model
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: OPENAI_API_KEY not found in environment variables")
            return None
            
        llm = ChatOpenAI(
            model="gpt-4o", 
            temperature=0.2,
            api_key=api_key
        )
        
        # Initialize connector
        connector = ComplianceCrewConnector(llm=llm)
        
        # Register all crews
        print("Registering compliance crews in the knowledge graph...")
        connector.register_all_compliance_crews()
        
        return connector
        
    except Exception as e:
        print(f"Error setting up compliance crews: {e}")
        return None


def demo_content_moderation(connector):
    """Demonstrate content moderation crew capabilities"""
    print("\n--- Content Moderation Demo ---")
    
    # Example task: Moderate a product description
    task_description = """
    Review and moderate the following product description for a children's toy:
    
    "Super Bubble Blaster 3000 - The most AMAZING bubble gun ever! Shoots bubbles up to 50 FEET! 
    Guaranteed to be the BEST bubble toy your kids will EVER use or DOUBLE YOUR MONEY BACK! 
    Made with high-quality materials that are COMPLETELY safe for children of ALL ages. 
    No other bubble gun on the market comes CLOSE to our performance. Get yours before they SELL OUT!"
    
    Identify any policy violations, misleading claims, and provide recommendations for changes.
    """
    
    # Execute the task with the content moderation crew
    result = connector.execute_task_with_crew(
        process_id="RETAIL-COMPLIANCE-001-002",
        task_description=task_description
    )
    
    if result["success"]:
        print("Content moderation result:")
        print(result["result"])
    else:
        print(f"Error executing content moderation task: {result['error']}")


def demo_fraud_prevention(connector):
    """Demonstrate fraud prevention crew capabilities"""
    print("\n--- Fraud Prevention Demo ---")
    
    # Example task: Analyze suspicious transaction patterns
    task_description = """
    Analyze the following suspicious transaction patterns and provide recommendations:
    
    1. Customer account created yesterday has placed 5 high-value orders using 3 different credit cards
    2. Same shipping address used across 12 different customer accounts in the past week
    3. Multiple failed payment attempts followed by successful payment with a different card
    4. Orders placed with express shipping but customer repeatedly contacts support asking to change delivery address
    5. Unusual pattern of expensive item purchases followed by return requests citing "not as described"
    
    Identify the level of fraud risk for each pattern and suggest preventative measures.
    """
    
    # Execute the task with the fraud prevention crew
    result = connector.execute_task_with_crew(
        process_id="RETAIL-COMPLIANCE-001-004",
        task_description=task_description
    )
    
    if result["success"]:
        print("Fraud prevention result:")
        print(result["result"])
    else:
        print(f"Error executing fraud prevention task: {result['error']}")


def demo_product_safety(connector):
    """Demonstrate product safety compliance crew capabilities"""
    print("\n--- Product Safety Compliance Demo ---")
    
    # Example task: Evaluate product safety documentation
    task_description = """
    Review the following product safety information for a new electronic toy for children ages 6-12:
    
    - Product contains small batteries
    - Has received CE certification for the European market
    - Contains magnets with flux index >50 kG²mm²
    - Manufacturer claims compliance with ASTM F963
    - Product has undergone flammability testing
    - Warning labels are in English only
    - No choking hazard testing documentation provided
    
    Identify any safety compliance issues, required certifications that may be missing, 
    and provide recommendations for addressing these issues before product launch.
    """
    
    # Execute the task with the product safety crew
    result = connector.execute_task_with_crew(
        process_id="RETAIL-COMPLIANCE-001-001",
        task_description=task_description
    )
    
    if result["success"]:
        print("Product safety compliance result:")
        print(result["result"])
    else:
        print(f"Error executing product safety task: {result['error']}")


def demo_cross_process_capability(connector):
    """Demonstrate finding agents with specific capabilities across processes"""
    print("\n--- Cross-Process Capability Demo ---")
    
    # Find agents with specific capabilities
    capabilities = [
        "Regulatory compliance verification",
        "Documentation",
        "Stakeholder communication"
    ]
    
    for capability in capabilities:
        agents = connector.find_agent_for_capability(capability)
        print(f"\nAgents with '{capability}' capability:")
        for agent in agents:
            print(f"  - {agent['name']} (ID: {agent['agent_id']}, Role: {agent['role']})")
            print(f"    Member of crews: {', '.join(agent['crews']) if agent['crews'] else 'None'}")


def main():
    """Main function to run the compliance crew demo"""
    print("Starting Compliance Crew Demo...")
    
    # Setup compliance crews
    connector = setup_compliance_crews()
    if not connector:
        print("Failed to set up compliance crews. Exiting.")
        return
    
    try:
        # Demo various compliance processes
        demo_content_moderation(connector)
        demo_fraud_prevention(connector)
        demo_product_safety(connector)
        
        # Demo cross-process capabilities
        demo_cross_process_capability(connector)
        
    except Exception as e:
        print(f"Error during demo: {e}")
    
    finally:
        # Close connections
        connector.close()
        print("\nCompliance Crew Demo completed!")


if __name__ == "__main__":
    main()