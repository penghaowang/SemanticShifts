"""
存储不同的prompt模板配置
"""

PROMPT_TEMPLATES = {
    # basic
    "basic": [
        {
            'role': 'system',
            'content': 'You are a helpful assistant. Only provide a concise definition or meaning of the target word, with no additional text, steps, explanations, or remarks.'
        },
        {
            'role': 'user',
            'content': "Sentence: '{sentence}'. Define the word '{word}' as used in this sentence. Provide ONLY the definition, without any extra words or phrases."
        },
        {
            'role': 'assistant',
            'content': ''
        }
    ],
    
    # icl_basic
    "icl_basic": [
        {
            'role': 'system',
            'content': 'You are a dictionary. Your only task is to provide a single definition by its context within 20 words. Do not include bullet points, steps, explanations, or any text besides the definition itself.'
        },
        {
            'role': 'user',
            'content': '''Examples:

Example 1:
Sentence: "The company's revenue showed significant growth in Q4."
Word: "growth"
 An increase in size, quantity, or importance over time.

Example 2:
Sentence: "The bank maintains strict policies regarding loan approval."
Word: "policies"
 Established rules and guidelines that govern decisions and procedures.

For the sentence: "{sentence}"
Define the word "{word}":'''
        },
        {
            'role': 'assistant',
            'content': ''
        }
    ],
    
    # icl_detailed
    "icl_detailed": [
        {
            'role': 'system',
            'content': 'You are a helpful assistant. Follow the examples to provide detailed word definitions with context analysis.'
        },
        {
            'role': 'user',
            'content': '''Here are examples of detailed word analysis in context:

Example 1:
Sentence: "The market volatility affected investment decisions."
Word: "volatility"
Analysis:
1. Domain: Finance/Investment
2. Context: Market behavior and risk assessment
3. Definition: The degree of variation in trading prices over time
4. Usage: Describes unpredictable changes in market conditions

Example 2:
Sentence: "The new regulation requires enhanced compliance measures."
Word: "compliance"
Analysis:
1. Domain: Legal/Regulatory
2. Context: Business operations and regulations
3. Definition: Adherence to rules, regulations, and standards
4. Usage: Indicates following mandatory requirements

Now, for the sentence: "{sentence}"
Analyze the word "{word}" following the same pattern.'''
        },
        {
            'role': 'assistant',
            'content': ''
        }
    ],
    'icl_context_aware': [
        {
            'role': 'system',
            'content': 'You are a helpful assistant. Follow the examples to provide context-sensitive word definitions that emphasize the specific usage in the given context.'
        },
        {
            'role': 'user',
            'content': '''Here are some examples:

Input: For sentence: "The company reported a significant gain in market share during the fourth quarter." Provide a context-sensitive definition for "gain:NOUN". Provide ONLY the definition, without any extra words or phrases.
Output: An increase in possession or acquisition, specifically referring to the expansion of company's market presence and business advantage.

Input: For sentence: "The investment portfolio showed substantial gains despite market volatility." Provide a context-sensitive definition for "gain:NOUN". Provide ONLY the definition, without any extra words or phrases.
Output: A positive financial return or profit obtained from investments, particularly in the context of market performance.

Input: For sentence: "The merger resulted in efficiency gains across all departments." Provide a context-sensitive definition for "gain:NOUN". Provide ONLY the definition, without any extra words or phrases.
Output: Improvements or benefits achieved, specifically referring to increased operational effectiveness resulting from business restructuring.

Input: For sentence: "{sentence}" Provide a context-sensitive definition for "{word}". Provide ONLY the definition, without any extra words or phrases.
Output:'''
        },
        {
            'role': 'assistant',
            'content': 'Output: '
        }
    ],
    'wordNet': [ # by giving wordnet definitions, request llm to generate the definition according to the wordnet definition
        {
            'role': 'system',
            'content': 'You are a helpful assistant. Follow the examples to provide detailed word definitions with context analysis.'
        },
        {
            'role': 'user',
            'content': '''For the wordnet definition: "{wordnet_definition}"
Generate a definition for the word "{word}" following the same pattern.'''
        },
        {
            'role': 'assistant',
            'content': ''
        }
    ]
} 
