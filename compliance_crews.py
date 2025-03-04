"""
Compliance Crews Module for AgentKG

This module defines CrewAI crews for handling various compliance processes
in the Retail domain, specifically for Trust and Safety operations.
"""

from crewai import Agent, Task, Crew
from crewai.tools.file_tools import FileReadTool, FileWriteTool
from langchain.tools import BaseTool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.language_models.chat_models import BaseChatModel
from typing import Optional, List, Dict, Any, Callable


class ComplianceCrews:
    """
    Factory class for creating compliance-focused crews.
    Each crew is specialized for a specific compliance process area.
    """
    
    def __init__(self, llm: Optional[BaseChatModel] = None):
        """
        Initialize the ComplianceCrews factory.
        
        Args:
            llm: Language model to use for the agents
        """
        self.llm = llm
        self.file_read_tool = FileReadTool()
        self.file_write_tool = FileWriteTool()
        self.wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    
    def create_content_moderation_crew(self) -> Crew:
        """
        Creates a crew specialized in content moderation processes.
        Maps to RETAIL-COMPLIANCE-001-002 in the knowledge graph.
        
        The crew includes agents for:
        - Product description review
        - Image moderation
        - Review moderation
        - Prohibited content detection
        - Appeals processing
        
        Returns:
            A CrewAI Crew object configured for content moderation
        """
        # Define specialized agents
        description_reviewer = Agent(
            role="Product Description Reviewer",
            goal="Ensure product descriptions comply with all company policies and regulations",
            backstory="""You are an expert in reviewing product descriptions to ensure they comply 
            with company policies, regulatory requirements, and best practices. You have extensive 
            experience identifying misleading claims, inappropriate content, and policy violations.""",
            verbose=True,
            llm=self.llm,
            tools=[self.file_read_tool, self.file_write_tool, self.wikipedia_tool]
        )
        
        image_moderator = Agent(
            role="Product Image Moderator",
            goal="Ensure all product images comply with company guidelines and are appropriate",
            backstory="""You specialize in reviewing product imagery to ensure compliance with company 
            policies. You can identify inappropriate content, misleading visuals, copyright issues, 
            and other policy violations in product images.""",
            verbose=True,
            llm=self.llm,
            tools=[self.file_read_tool, self.file_write_tool]
        )
        
        review_moderator = Agent(
            role="Customer Review Moderator",
            goal="Ensure customer reviews and Q&A content follows community guidelines",
            backstory="""You are skilled at moderating user-generated content like reviews and questions. 
            You can identify inappropriate language, misleading information, spam, and content that 
            violates community guidelines while preserving authentic customer feedback.""",
            verbose=True,
            llm=self.llm,
            tools=[self.file_read_tool, self.file_write_tool]
        )
        
        content_detector = Agent(
            role="Prohibited Content Detector",
            goal="Identify and flag prohibited or illegal content across all channels",
            backstory="""You are an expert in detecting prohibited content across all retail channels. 
            You have deep knowledge of regulations around restricted products, illegal items, and 
            content that violates marketplace policies. You can detect subtle attempts to circumvent 
            detection systems.""",
            verbose=True,
            llm=self.llm,
            tools=[self.file_read_tool, self.file_write_tool, self.wikipedia_tool]
        )
        
        appeals_processor = Agent(
            role="Moderation Appeals Processor",
            goal="Fairly review appeals against moderation decisions",
            backstory="""You specialize in reviewing appeals against content moderation decisions. 
            You can objectively evaluate if the original moderation decision was correct based on 
            policies and context. You understand the balance between strict policy enforcement and 
            fairness to sellers.""",
            verbose=True,
            llm=self.llm,
            tools=[self.file_read_tool, self.file_write_tool]
        )
        
        # Define tasks for each agent
        description_review_task = Task(
            description="""
            Review product descriptions to identify policy violations:
            1. Check for misleading claims and false advertising
            2. Verify compliance with regulatory requirements
            3. Screen for inappropriate content or terminology
            4. Ensure consistency with product categorization
            5. Generate a detailed compliance report with specific recommendations
            """,
            agent=description_reviewer,
            expected_output="A compliance report on product descriptions with specific violations and recommendations"
        )
        
        image_moderation_task = Task(
            description="""
            Review product images to identify policy violations:
            1. Check for inappropriate or offensive content
            2. Verify that images accurately represent the product
            3. Identify any copyright or trademark violations
            4. Ensure compliance with image quality standards
            5. Generate a detailed report with specific recommendations
            """,
            agent=image_moderator,
            expected_output="A compliance report on product images with specific violations and recommendations"
        )
        
        review_moderation_task = Task(
            description="""
            Moderate customer reviews and Q&A content:
            1. Identify and flag inappropriate language or personal attacks
            2. Detect spam or fake reviews
            3. Remove personally identifiable information
            4. Check for compliance with review guidelines
            5. Generate a moderation report summarizing actions taken
            """,
            agent=review_moderator,
            expected_output="A moderation report on customer reviews and Q&A content with actions taken"
        )
        
        prohibited_content_task = Task(
            description="""
            Perform deep scanning for prohibited content:
            1. Scan product listings for illegal items or services
            2. Identify subtle attempts to circumvent detection systems
            3. Check for regulatory compliance issues
            4. Cross-reference with known patterns of prohibited content
            5. Generate an alert report for all detected violations
            """,
            agent=content_detector,
            expected_output="A comprehensive alert report on prohibited content with severity rankings"
        )
        
        appeals_processing_task = Task(
            description="""
            Process appeals against moderation decisions:
            1. Review the original moderation decision and rationale
            2. Examine the appeal justification from the seller/customer
            3. Research relevant policies and precedents
            4. Make a fair determination based on all evidence
            5. Provide a detailed justification for the appeal decision
            """,
            agent=appeals_processor,
            expected_output="A decision report on moderation appeals with detailed justifications"
        )
        
        # Create the crew
        content_moderation_crew = Crew(
            agents=[description_reviewer, image_moderator, review_moderator, content_detector, appeals_processor],
            tasks=[description_review_task, image_moderation_task, review_moderation_task, prohibited_content_task, appeals_processing_task],
            verbose=2,
            process=Crew.SEQUENTIAL
        )
        
        return content_moderation_crew

    def create_fraud_prevention_crew(self) -> Crew:
        """
        Creates a crew specialized in fraud prevention processes.
        Maps to RETAIL-COMPLIANCE-001-004 in the knowledge graph.
        
        The crew includes agents for:
        - Fraud detection
        - Fraud investigation
        - Fraud mitigation
        - Fraud reporting
        
        Returns:
            A CrewAI Crew object configured for fraud prevention
        """
        # Define specialized agents
        fraud_detector = Agent(
            role="Fraud Detection Specialist",
            goal="Identify patterns and indicators of potential fraud in retail operations",
            backstory="""You are an expert in detecting fraudulent activities in retail environments. 
            You have extensive experience with pattern recognition, anomaly detection, and fraud 
            signals. You can identify unusual behaviors that might indicate fraud across ordering, 
            returns, payments, and account activities.""",
            verbose=True,
            llm=self.llm,
            tools=[self.file_read_tool, self.file_write_tool]
        )
        
        fraud_investigator = Agent(
            role="Fraud Investigator",
            goal="Thoroughly investigate potential fraud cases to confirm or dismiss concerns",
            backstory="""You specialize in investigating suspected fraud cases. You have a background 
            in forensic analysis and can follow evidence trails across multiple systems to confirm 
            whether fraud has occurred. You know how to document findings in a way that meets legal 
            and compliance standards.""",
            verbose=True,
            llm=self.llm,
            tools=[self.file_read_tool, self.file_write_tool]
        )
        
        fraud_mitigator = Agent(
            role="Fraud Mitigation Expert",
            goal="Implement measures to prevent and reduce fraud incidents",
            backstory="""You are skilled at developing and implementing strategies to prevent and 
            mitigate fraud. You understand how to balance security measures with customer experience, 
            and can design systems that reduce fraud without creating excessive friction for legitimate 
            customers.""",
            verbose=True,
            llm=self.llm,
            tools=[self.file_read_tool, self.file_write_tool]
        )
        
        fraud_reporter = Agent(
            role="Fraud Analytics and Reporting Specialist",
            goal="Analyze fraud trends and create comprehensive reports for stakeholders",
            backstory="""You excel at analyzing fraud data and creating insightful reports. You can 
            identify emerging fraud patterns, measure the effectiveness of prevention strategies, 
            and communicate complex fraud information in a way that helps stakeholders make informed 
            decisions.""",
            verbose=True,
            llm=self.llm,
            tools=[self.file_read_tool, self.file_write_tool]
        )
        
        # Define tasks for each agent
        detection_task = Task(
            description="""
            Analyze transaction and activity data to detect potential fraud:
            1. Scan for unusual patterns in transaction data
            2. Identify suspicious account activities
            3. Flag anomalies in ordering and shipping behaviors
            4. Detect potential policy abuse patterns
            5. Generate a list of potential fraud cases with confidence scores
            """,
            agent=fraud_detector,
            expected_output="A prioritized list of potential fraud cases with supporting evidence and confidence scores"
        )
        
        investigation_task = Task(
            description="""
            Investigate flagged potential fraud cases:
            1. Analyze all available evidence for each flagged case
            2. Follow transaction trails across systems
            3. Document findings in detail according to compliance standards
            4. Make determinations on whether fraud has occurred
            5. Prepare comprehensive investigation reports
            """,
            agent=fraud_investigator,
            expected_output="Detailed investigation reports for each case with conclusions and evidence summaries"
        )
        
        mitigation_task = Task(
            description="""
            Develop and implement fraud prevention strategies:
            1. Analyze confirmed fraud cases to identify vulnerability patterns
            2. Design preventative measures and controls
            3. Develop rule sets for automated fraud detection
            4. Create security protocols that balance protection with customer experience
            5. Generate a comprehensive fraud mitigation plan
            """,
            agent=fraud_mitigator,
            expected_output="A detailed fraud mitigation plan with specific measures and implementation guidelines"
        )
        
        reporting_task = Task(
            description="""
            Analyze fraud trends and create reports:
            1. Consolidate data from all fraud cases
            2. Identify emerging patterns and trends
            3. Evaluate the effectiveness of current prevention measures
            4. Calculate financial impact of fraud activities
            5. Create comprehensive reports for different stakeholder groups
            """,
            agent=fraud_reporter,
            expected_output="A comprehensive fraud trend analysis with visualizations and recommendations"
        )
        
        # Create the crew
        fraud_prevention_crew = Crew(
            agents=[fraud_detector, fraud_investigator, fraud_mitigator, fraud_reporter],
            tasks=[detection_task, investigation_task, mitigation_task, reporting_task],
            verbose=2,
            process=Crew.SEQUENTIAL
        )
        
        return fraud_prevention_crew

    def create_product_safety_compliance_crew(self) -> Crew:
        """
        Creates a crew specialized in product safety compliance processes.
        Maps to RETAIL-COMPLIANCE-001-001 in the knowledge graph.
        
        The crew includes agents for:
        - Safety certification verification
        - Product recall management
        - Product safety testing
        - Safety compliance reporting
        
        Returns:
            A CrewAI Crew object configured for product safety compliance
        """
        # Define specialized agents
        certification_verifier = Agent(
            role="Safety Certification Verifier",
            goal="Verify product safety certifications and documentation for compliance",
            backstory="""You are an expert in product safety certifications across different industries 
            and regions. You know which certifications are required for different product types and 
            can verify the authenticity and validity of safety documentation. You understand global 
            safety standards and can identify when products need additional certification.""",
            verbose=True,
            llm=self.llm,
            tools=[self.file_read_tool, self.file_write_tool, self.wikipedia_tool]
        )
        
        recall_manager = Agent(
            role="Product Recall Manager",
            goal="Effectively manage product recalls and safety alerts",
            backstory="""You specialize in managing product recalls and safety alerts. You understand 
            the regulatory requirements around recalls, know how to coordinate with manufacturers, 
            distribution centers, and customers, and can design effective recall communication strategies 
            that minimize brand damage while maximizing customer safety.""",
            verbose=True,
            llm=self.llm,
            tools=[self.file_read_tool, self.file_write_tool]
        )
        
        safety_tester = Agent(
            role="Product Safety Testing Specialist",
            goal="Ensure products are tested thoroughly for safety compliance",
            backstory="""You have extensive experience in product safety testing protocols and standards. 
            You understand different testing methodologies for various product categories, can interpret 
            test results accurately, and know when additional testing is required. You stay current with 
            evolving safety testing requirements across different markets.""",
            verbose=True,
            llm=self.llm,
            tools=[self.file_read_tool, self.file_write_tool]
        )
        
        safety_reporter = Agent(
            role="Safety Compliance Reporting Specialist",
            goal="Create comprehensive safety compliance reports for stakeholders",
            backstory="""You excel at analyzing safety compliance data and creating detailed reports 
            that meet regulatory requirements. You can clearly communicate complex safety information 
            to different stakeholders, from executives to regulatory bodies. You understand what needs 
            to be documented and reported to maintain compliance.""",
            verbose=True,
            llm=self.llm,
            tools=[self.file_read_tool, self.file_write_tool]
        )
        
        # Define tasks for each agent
        certification_task = Task(
            description="""
            Verify product safety certifications:
            1. Review product specifications to determine required certifications
            2. Validate submitted certification documents for authenticity
            3. Check certification validity and expiration dates
            4. Identify missing or incomplete certifications
            5. Prepare a certification compliance report
            """,
            agent=certification_verifier,
            expected_output="A detailed certification compliance report identifying valid, expired, and missing certifications"
        )
        
        recall_task = Task(
            description="""
            Design and implement a product recall process:
            1. Create a recall action plan based on safety issue severity
            2. Develop communication templates for different stakeholders
            3. Establish protocols for product return and disposition
            4. Design tracking mechanisms for recall effectiveness
            5. Prepare a comprehensive recall management playbook
            """,
            agent=recall_manager,
            expected_output="A complete product recall management playbook with templates and procedures"
        )
        
        testing_task = Task(
            description="""
            Develop safety testing protocols:
            1. Identify required safety tests for different product categories
            2. Create testing checklists and procedures
            3. Establish criteria for pass/fail determinations
            4. Design protocols for handling testing failures
            5. Generate comprehensive testing documentation templates
            """,
            agent=safety_tester,
            expected_output="A set of safety testing protocols with specific procedures for different product categories"
        )
        
        reporting_task = Task(
            description="""
            Create safety compliance reporting systems:
            1. Design templates for different types of safety reports
            2. Establish reporting frequencies and triggers
            3. Identify key performance indicators for safety compliance
            4. Create dashboard designs for monitoring safety metrics
            5. Develop a comprehensive safety reporting framework
            """,
            agent=safety_reporter,
            expected_output="A complete safety compliance reporting framework with templates and dashboards"
        )
        
        # Create the crew
        product_safety_crew = Crew(
            agents=[certification_verifier, recall_manager, safety_tester, safety_reporter],
            tasks=[certification_task, recall_task, testing_task, reporting_task],
            verbose=2,
            process=Crew.SEQUENTIAL
        )
        
        return product_safety_crew


# Example usage:
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model="gpt-4o")
# compliance_crews = ComplianceCrews(llm=llm)
# content_crew = compliance_crews.create_content_moderation_crew()
# result = content_crew.kickoff()