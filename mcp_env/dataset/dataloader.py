import json
import os
from datasets import load_dataset
from typing import List, Dict, Any, Optional
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

class TaskBenchToMCPConverter:
    """Download Microsoft TaskBench dataset and convert to Model Context Protocol format"""
    
    def __init__(self, output_dir: str = "./taskbench_mcp_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.repo_id = "microsoft/Taskbench"
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🎯 Target dataset: {self.repo_id}")
    
    def download_taskbench_data(self) -> Dict[str, List[Dict]]:
        """Download Microsoft TaskBench dataset with proper config handling"""
        print("🔄 Downloading Microsoft TaskBench dataset...")
        
        # TaskBench has specific configurations
        taskbench_configs = [
            {
                "name": "huggingface", 
                "description": "HuggingFace-related tasks and tools"
            },
            {
                "name": "multimedia", 
                "description": "Multimedia processing and generation tasks"
            },
            {
                "name": "dailylifeapis", 
                "description": "Daily life and common API usage tasks"
            }
        ]
        
        all_data = {}
        successful_configs = 0
        
        for config in taskbench_configs:
            config_name = config["name"]
            try:
                print(f"\n📊 Loading TaskBench config '{config_name}'...")
                print(f"   📝 {config['description']}")
                
                dataset = load_dataset(self.repo_id, config_name, trust_remote_code=True)
                
                print(f"✅ Successfully loaded '{config_name}' config")
                print(f"📋 Available splits: {list(dataset.keys())}")
                
                # Process each split in this config
                for split_name in dataset.keys():
                    try:
                        split_data = dataset[split_name]
                        print(f"   📊 Processing '{config_name}' split '{split_name}': {len(split_data)} examples")
                        
                        # Convert to list and limit for processing
                        processed_data = []
                        for i, item in enumerate(split_data):
                            if i >= 150:  # Limit per config-split combination
                                break
                            processed_data.append(dict(item))
                        
                        # Use config_split format for unique identification
                        split_key = f"{config_name}_{split_name}"
                        all_data[split_key] = processed_data
                        print(f"   ✅ Processed {len(processed_data)} examples from {split_key}")
                        
                    except Exception as e:
                        print(f"   ⚠️ Failed to process split {split_name} in config {config_name}: {e}")
                        continue
                
                successful_configs += 1
                
            except Exception as e:
                print(f"❌ Failed to load config '{config_name}': {e}")
                continue
        
        if successful_configs > 0:
            print(f"\n✅ Successfully loaded {successful_configs}/{len(taskbench_configs)} TaskBench configurations")
            return all_data
        else:
            print("❌ Failed to load any TaskBench configurations")
            print("🔄 Trying alternative approaches...")
            return self._try_alternative_loading()
    
    def _try_alternative_loading(self) -> Dict[str, List[Dict]]:
        """Try alternative loading methods for TaskBench with specific configs"""
        print("🔄 Trying alternative TaskBench loading methods...")
        
        # Try each config individually with error handling
        configs_to_try = ['huggingface', 'multimedia', 'dailylifeapis']
        
        all_data = {}
        
        for config in configs_to_try:
            try:
                print(f"   🔄 Attempting direct load of '{config}' config...")
                dataset = load_dataset("microsoft/Taskbench", config, trust_remote_code=True)
                
                for split_name in dataset.keys():
                    try:
                        split_data = dataset[split_name]
                        processed_data = [dict(item) for i, item in enumerate(split_data) if i < 100]
                        split_key = f"alt_{config}_{split_name}"
                        all_data[split_key] = processed_data
                        print(f"   ✅ Loaded {len(processed_data)} from {config}_{split_name}")
                    except Exception as e:
                        print(f"   ⚠️ Failed to process {config}_{split_name}: {e}")
                        continue
                        
            except Exception as e:
                print(f"   ⚠️ Config '{config}' failed: {e}")
                continue
        
        if all_data:
            print(f"✅ Alternative loading successful! Got data from configs.")
            return all_data
        
        # If all else fails, create enhanced TaskBench-style sample data
        print("⚠️ Using enhanced TaskBench-style sample data...")
        return self._create_enhanced_taskbench_sample_data()
    
    def _create_enhanced_taskbench_sample_data(self) -> Dict[str, List[Dict]]:
        """Create enhanced TaskBench-style sample data representing all three configs"""
        return {
            "huggingface_train": [
                {
                    "task_id": "hf_001",
                    "instruction": "I want to find a good text classification model on Hugging Face that can classify movie reviews as positive or negative. Can you help me search for suitable models?",
                    "tools": [
                        {
                            "name": "search_huggingface_models",
                            "description": "Search for models on Hugging Face Hub",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "task": {"type": "string", "description": "The task type (e.g., text-classification)"},
                                    "query": {"type": "string", "description": "Search query"},
                                    "sort": {"type": "string", "enum": ["downloads", "recent", "trending"], "description": "Sort order"},
                                    "filter": {"type": "string", "description": "Additional filters"}
                                },
                                "required": ["task"]
                            }
                        },
                        {
                            "name": "get_model_details",
                            "description": "Get detailed information about a specific model",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "model_id": {"type": "string", "description": "The model identifier"},
                                    "include_metrics": {"type": "boolean", "description": "Whether to include performance metrics"}
                                },
                                "required": ["model_id"]
                            }
                        }
                    ],
                    "solution": [
                        {
                            "step": 1,
                            "action": "search_huggingface_models",
                            "parameters": {
                                "task": "text-classification",
                                "query": "sentiment analysis movie reviews",
                                "sort": "downloads"
                            }
                        },
                        {
                            "step": 2,
                            "action": "get_model_details",
                            "parameters": {
                                "model_id": "cardiffnlp/twitter-roberta-base-sentiment-latest",
                                "include_metrics": True
                            }
                        }
                    ],
                    "domain": "machine_learning",
                    "difficulty": "medium",
                    "multi_step": True
                }
            ],
            "multimedia_train": [
                {
                    "task_id": "mm_001",
                    "instruction": "I have an image that I want to enhance and then generate a caption for it. Can you help me process this image?",
                    "tools": [
                        {
                            "name": "enhance_image",
                            "description": "Enhance image quality using AI upscaling",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "image_path": {"type": "string", "description": "Path to the input image"},
                                    "enhancement_type": {"type": "string", "enum": ["upscale", "denoise", "sharpen"], "description": "Type of enhancement"},
                                    "output_path": {"type": "string", "description": "Path for the enhanced image"}
                                },
                                "required": ["image_path", "enhancement_type"]
                            }
                        },
                        {
                            "name": "generate_image_caption",
                            "description": "Generate a descriptive caption for an image",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "image_path": {"type": "string", "description": "Path to the image"},
                                    "max_length": {"type": "integer", "description": "Maximum caption length"},
                                    "style": {"type": "string", "enum": ["descriptive", "creative", "technical"], "description": "Caption style"}
                                },
                                "required": ["image_path"]
                            }
                        }
                    ],
                    "solution": [
                        {
                            "step": 1,
                            "action": "enhance_image",
                            "parameters": {
                                "image_path": "/path/to/input.jpg",
                                "enhancement_type": "upscale",
                                "output_path": "/path/to/enhanced.jpg"
                            }
                        },
                        {
                            "step": 2,
                            "action": "generate_image_caption",
                            "parameters": {
                                "image_path": "/path/to/enhanced.jpg",
                                "max_length": 100,
                                "style": "descriptive"
                            }
                        }
                    ],
                    "domain": "multimedia",
                    "difficulty": "medium",
                    "multi_step": True
                }
            ],
            "dailylifeapis_train": [
                {
                    "task_id": "daily_001",
                    "instruction": "I want to check the weather for tomorrow and then set a reminder to bring an umbrella if it's going to rain.",
                    "tools": [
                        {
                            "name": "get_weather_forecast",
                            "description": "Get weather forecast for specified days",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "location": {"type": "string", "description": "City name or coordinates"},
                                    "days": {"type": "integer", "description": "Number of days to forecast"},
                                    "include_details": {"type": "boolean", "description": "Include detailed weather info"}
                                },
                                "required": ["location", "days"]
                            }
                        },
                        {
                            "name": "create_reminder",
                            "description": "Create a reminder notification",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string", "description": "Reminder title"},
                                    "message": {"type": "string", "description": "Reminder message"},
                                    "datetime": {"type": "string", "description": "When to trigger the reminder (ISO format)"},
                                    "priority": {"type": "string", "enum": ["low", "medium", "high"], "description": "Reminder priority"}
                                },
                                "required": ["title", "datetime"]
                            }
                        }
                    ],
                    "solution": [
                        {
                            "step": 1,
                            "action": "get_weather_forecast",
                            "parameters": {
                                "location": "user_location",
                                "days": 1,
                                "include_details": True
                            }
                        },
                        {
                            "step": 2,
                            "action": "create_reminder",
                            "parameters": {
                                "title": "Weather Reminder",
                                "message": "Don't forget to bring an umbrella - rain expected!",
                                "datetime": "2025-05-28T07:00:00Z",
                                "priority": "medium"
                            }
                        }
                    ],
                    "domain": "daily_life",
                    "difficulty": "simple",
                    "multi_step": True
                },
                {
                    "task_id": "daily_002",
                    "instruction": "Find a good restaurant nearby for dinner tonight and make a reservation.",
                    "tools": [
                        {
                            "name": "search_restaurants",
                            "description": "Search for restaurants in a specific area",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "location": {"type": "string", "description": "Search location"},
                                    "cuisine": {"type": "string", "description": "Type of cuisine"},
                                    "price_range": {"type": "string", "enum": ["$", "$", "$$", "$$"], "description": "Price range"},
                                    "rating_min": {"type": "number", "description": "Minimum rating"},
                                    "open_now": {"type": "boolean", "description": "Only show currently open restaurants"}
                                },
                                "required": ["location"]
                            }
                        },
                        {
                            "name": "make_reservation",
                            "description": "Make a restaurant reservation",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "restaurant_id": {"type": "string", "description": "Restaurant identifier"},
                                    "date": {"type": "string", "description": "Reservation date (YYYY-MM-DD)"},
                                    "time": {"type": "string", "description": "Reservation time (HH:MM)"},
                                    "party_size": {"type": "integer", "description": "Number of people"},
                                    "special_requests": {"type": "string", "description": "Any special requests"}
                                },
                                "required": ["restaurant_id", "date", "time", "party_size"]
                            }
                        }
                    ],
                    "solution": [
                        {
                            "step": 1,
                            "action": "search_restaurants",
                            "parameters": {
                                "location": "nearby",
                                "price_range": "$",
                                "rating_min": 4.0,
                                "open_now": False
                            }
                        },
                        {
                            "step": 2,
                            "action": "make_reservation",
                            "parameters": {
                                "restaurant_id": "selected_restaurant_id",
                                "date": "2025-05-27",
                                "time": "19:00",
                                "party_size": 2
                            }
                        }
                    ],
                    "domain": "daily_life",
                    "difficulty": "medium",
                    "multi_step": True
                }
            ]
        }
        """Create TaskBench-style sample data based on Microsoft's format"""
        return {
            "train": [
                {
                    "task_id": "task_001",
                    "instruction": "I need to schedule a meeting with my team for next Tuesday at 2 PM. Can you help me set this up and send calendar invites?",
                    "tools": [
                        {
                            "name": "create_calendar_event",
                            "description": "Create a new calendar event",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string", "description": "Event title"},
                                    "date": {"type": "string", "description": "Event date (YYYY-MM-DD)"},
                                    "time": {"type": "string", "description": "Event time (HH:MM)"},
                                    "duration": {"type": "integer", "description": "Duration in minutes"},
                                    "attendees": {"type": "array", "items": {"type": "string"}, "description": "List of attendee emails"}
                                },
                                "required": ["title", "date", "time"]
                            }
                        },
                        {
                            "name": "send_calendar_invite",
                            "description": "Send calendar invitations to attendees",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "event_id": {"type": "string", "description": "ID of the calendar event"},
                                    "message": {"type": "string", "description": "Optional message to include with invite"}
                                },
                                "required": ["event_id"]
                            }
                        }
                    ],
                    "solution": [
                        {
                            "step": 1,
                            "action": "create_calendar_event",
                            "parameters": {
                                "title": "Team Meeting",
                                "date": "2025-06-03",
                                "time": "14:00",
                                "duration": 60,
                                "attendees": ["team@company.com"]
                            }
                        },
                        {
                            "step": 2,
                            "action": "send_calendar_invite",
                            "parameters": {
                                "event_id": "generated_event_id",
                                "message": "Team meeting to discuss project updates"
                            }
                        }
                    ],
                    "domain": "productivity",
                    "difficulty": "medium",
                    "multi_step": True
                },
                {
                    "task_id": "task_002",
                    "instruction": "What's the current weather in Seattle?",
                    "tools": [
                        {
                            "name": "get_weather",
                            "description": "Get current weather information for a city",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "city": {"type": "string", "description": "City name"},
                                    "country": {"type": "string", "description": "Country code (optional)"},
                                    "units": {"type": "string", "enum": ["metric", "imperial"], "description": "Temperature units"}
                                },
                                "required": ["city"]
                            }
                        }
                    ],
                    "solution": [
                        {
                            "step": 1,
                            "action": "get_weather",
                            "parameters": {
                                "city": "Seattle",
                                "country": "US",
                                "units": "imperial"
                            }
                        }
                    ],
                    "domain": "information",
                    "difficulty": "simple",
                    "multi_step": False
                },
                {
                    "task_id": "task_003",
                    "instruction": "I want to book a flight from New York to London, find hotels in London, and get the weather forecast for my travel dates.",
                    "tools": [
                        {
                            "name": "search_flights",
                            "description": "Search for available flights",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "origin": {"type": "string", "description": "Departure city"},
                                    "destination": {"type": "string", "description": "Arrival city"},
                                    "departure_date": {"type": "string", "description": "Departure date (YYYY-MM-DD)"},
                                    "return_date": {"type": "string", "description": "Return date (YYYY-MM-DD)"},
                                    "passengers": {"type": "integer", "description": "Number of passengers"}
                                },
                                "required": ["origin", "destination", "departure_date"]
                            }
                        },
                        {
                            "name": "search_hotels",
                            "description": "Search for hotels in a city",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "city": {"type": "string", "description": "City name"},
                                    "checkin_date": {"type": "string", "description": "Check-in date (YYYY-MM-DD)"},
                                    "checkout_date": {"type": "string", "description": "Check-out date (YYYY-MM-DD)"},
                                    "guests": {"type": "integer", "description": "Number of guests"},
                                    "min_rating": {"type": "number", "description": "Minimum hotel rating"}
                                },
                                "required": ["city", "checkin_date", "checkout_date"]
                            }
                        },
                        {
                            "name": "get_weather_forecast",
                            "description": "Get weather forecast for multiple days",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "city": {"type": "string", "description": "City name"},
                                    "start_date": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                                    "end_date": {"type": "string", "description": "End date (YYYY-MM-DD)"}
                                },
                                "required": ["city", "start_date", "end_date"]
                            }
                        }
                    ],
                    "solution": [
                        {
                            "step": 1,
                            "action": "search_flights",
                            "parameters": {
                                "origin": "New York",
                                "destination": "London",
                                "departure_date": "2025-07-15",
                                "return_date": "2025-07-22",
                                "passengers": 1
                            }
                        },
                        {
                            "step": 2,
                            "action": "search_hotels",
                            "parameters": {
                                "city": "London",
                                "checkin_date": "2025-07-15",
                                "checkout_date": "2025-07-22",
                                "guests": 1,
                                "min_rating": 4.0
                            }
                        },
                        {
                            "step": 3,
                            "action": "get_weather_forecast",
                            "parameters": {
                                "city": "London",
                                "start_date": "2025-07-15",
                                "end_date": "2025-07-22"
                            }
                        }
                    ],
                    "domain": "travel",
                    "difficulty": "complex",
                    "multi_step": True
                }
            ],
            "test": [
                {
                    "task_id": "test_001",
                    "instruction": "Send an email to john@example.com with the subject 'Project Update' and ask him about the status of the quarterly report.",
                    "tools": [
                        {
                            "name": "send_email",
                            "description": "Send an email message",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "to": {"type": "string", "description": "Recipient email address"},
                                    "subject": {"type": "string", "description": "Email subject"},
                                    "body": {"type": "string", "description": "Email body content"},
                                    "cc": {"type": "array", "items": {"type": "string"}, "description": "CC recipients"},
                                    "priority": {"type": "string", "enum": ["low", "normal", "high"], "description": "Email priority"}
                                },
                                "required": ["to", "subject", "body"]
                            }
                        }
                    ],
                    "solution": [
                        {
                            "step": 1,
                            "action": "send_email",
                            "parameters": {
                                "to": "john@example.com",
                                "subject": "Project Update",
                                "body": "Hi John, I hope you're doing well. Could you please provide an update on the status of the quarterly report? Thanks!",
                                "priority": "normal"
                            }
                        }
                    ],
                    "domain": "communication",
                    "difficulty": "simple",
                    "multi_step": False
                }
            ]
        }
    
    def convert_to_mcp_format(self, taskbench_data: Dict[str, List[Dict]]) -> List[Dict[str, Any]]:
        """Convert TaskBench format to Model Context Protocol format"""
        print("\n🔄 Converting TaskBench to MCP format...")
        
        mcp_data = []
        
        for split_name, split_data in taskbench_data.items():
            print(f"📊 Converting split '{split_name}': {len(split_data)} examples")
            
            for i, item in enumerate(split_data):
                try:
                    mcp_example = self._convert_single_taskbench_example(item, split_name, i)
                    if mcp_example:
                        mcp_data.append(mcp_example)
                except Exception as e:
                    print(f"   ⚠️ Failed to convert example {i}: {e}")
                    continue
        
        print(f"✅ Converted {len(mcp_data)} examples to MCP format")
        return mcp_data
    
    def _convert_single_taskbench_example(self, item: Dict, split_name: str, index: int) -> Optional[Dict[str, Any]]:
        """Convert a single TaskBench example to MCP format"""
        
        # Debug: Print first few items to understand structure
        if index < 3:
            print(f"   🔍 Example {index} fields: {list(item.keys())}")
            print(f"   🔍 Sample content: {str(item)[:200]}...")
        
        # TaskBench uses 'instruction' as the main query field
        query_fields = ['instruction', 'query', 'task', 'prompt', 'question']
        query = None
        for field in query_fields:
            if field in item and item[field]:
                query = item[field]
                break
        
        if not query:
            if index < 3:
                print(f"   ⚠️ No query found in: {item}")
            return None
        
        # TaskBench stores tools in 'sampled_nodes' field as JSON string
        tools = []
        
        # Try to extract tools from TaskBench format
        if 'sampled_nodes' in item:
            try:
                # Parse the JSON string in sampled_nodes
                sampled_nodes = json.loads(item['sampled_nodes']) if isinstance(item['sampled_nodes'], str) else item['sampled_nodes']
                
                if isinstance(sampled_nodes, list):
                    for node in sampled_nodes:
                        if isinstance(node, dict) and 'task' in node:
                            # Convert TaskBench node to MCP tool format
                            tool = {
                                "name": node['task'],
                                "description": f"Task: {node['task']}",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {},
                                    "required": []
                                }
                            }
                            
                            # Process arguments if available
                            if 'arguments' in node and isinstance(node['arguments'], list):
                                for arg in node['arguments']:
                                    if isinstance(arg, dict) and 'name' in arg:
                                        prop_name = arg['name']
                                        tool["inputSchema"]["properties"][prop_name] = {
                                            "type": arg.get('type', 'string'),
                                            "description": arg.get('desc', f"Parameter {prop_name}")
                                        }
                                        # All TaskBench arguments appear to be required based on examples
                                        tool["inputSchema"]["required"].append(prop_name)
                            
                            tools.append(tool)
                            
            except (json.JSONDecodeError, TypeError) as e:
                if index < 3:
                    print(f"   ⚠️ Failed to parse sampled_nodes: {e}")
        
        # If no tools found, try alternative fields
        if not tools:
            for field in ['tools', 'functions', 'apis', 'available_tools']:
                if field in item and item[field]:
                    if index < 3:
                        print(f"   🔍 Found {field}: {item[field]}")
                    # Process alternative tool formats
                    break
        
        if not tools:
            if index < 3:
                print(f"   ⚠️ No tools found in example {index}")
            return None
        
        # Create MCP example
        mcp_example = {
            "id": f"{split_name}_{index}",
            "query": query,
            "tools": tools,
            "metadata": {
                "source": "Microsoft_TaskBench",
                "split": split_name,
                "original_index": index,
                "num_tools": len(tools),
                "config": self._extract_config_from_split(split_name),
                "task_type": item.get('type', 'unknown'),  # TaskBench has 'type' field
                "n_tools": item.get('n_tools', len(tools)),  # TaskBench has 'n_tools' field
                "seed": item.get('seed', None)  # TaskBench has 'seed' field
            }
        }
        
        # Add TaskBench specific fields
        if 'tool_steps' in item:
            mcp_example['expected_steps'] = item['tool_steps']
        
        if 'tool_nodes' in item:
            mcp_example['expected_nodes'] = item['tool_nodes']
        
        if 'sampled_links' in item:
            mcp_example['expected_links'] = item['sampled_links']
        
        return mcp_example
    
    def _extract_config_from_split(self, split_name: str) -> str:
        """Extract TaskBench config from split name"""
        if 'huggingface' in split_name.lower():
            return 'huggingface'
        elif 'multimedia' in split_name.lower():
            return 'multimedia'  
        elif 'dailylife' in split_name.lower():
            return 'dailylifeapis'
        else:
            return 'unknown'
    
    def _determine_task_type(self, item: Dict, tools: List[Dict]) -> str:
        """Determine the type of task based on content and tools"""
        
        # Check for explicit task type
        if 'task_type' in item:
            return item['task_type']
        
        # Infer from domain
        domain = item.get('domain', '').lower()
        if domain in ['travel', 'booking']:
            return 'planning'
        elif domain in ['communication', 'email']:
            return 'communication'
        elif domain in ['productivity', 'calendar']:
            return 'productivity'
        elif domain in ['information', 'search']:
            return 'information_retrieval'
        
        # Infer from tools
        tool_names = [tool.get('name', '').lower() for tool in tools]
        
        if any('email' in name or 'message' in name for name in tool_names):
            return 'communication'
        elif any('calendar' in name or 'schedule' in name for name in tool_names):
            return 'scheduling'
        elif any('search' in name or 'find' in name for name in tool_names):
            return 'search'
        elif any('weather' in name for name in tool_names):
            return 'information_retrieval'
        
        # Default based on complexity
        if len(tools) > 2:
            return 'complex_workflow'
        elif len(tools) == 1:
            return 'simple_action'
        else:
            return 'multi_step'
    
    def analyze_dataset(self, mcp_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the converted TaskBench MCP dataset"""
        print("\n📊 Analyzing TaskBench MCP dataset...")
        
        if not mcp_data:
            return {}
        
        # Collect comprehensive statistics
        total_examples = len(mcp_data)
        tool_names = []
        tools_per_example = []
        task_types = []
        splits = []
        configs = []  # Add config tracking
        query_lengths = []
        seeds = []
        n_tools_original = []
        
        for example in mcp_data:
            # Tools analysis
            example_tools = example.get('tools', [])
            tools_per_example.append(len(example_tools))
            tool_names.extend([tool.get('name', 'unknown') for tool in example_tools])
            
            # Query analysis
            query = example.get('query', '')
            query_lengths.append(len(query.split()) if query else 0)
            
            # Metadata analysis
            metadata = example.get('metadata', {})
            task_types.append(metadata.get('task_type', 'unknown'))
            splits.append(metadata.get('split', 'unknown'))
            configs.append(metadata.get('config', 'unknown'))  # Track configs
            
            # TaskBench specific fields
            if 'seed' in metadata and metadata['seed']:
                seeds.append(metadata['seed'])
            if 'n_tools' in metadata and metadata['n_tools']:
                n_tools_original.append(metadata['n_tools'])
        
        # Calculate statistics
        from collections import Counter
        
        def safe_stats(values):
            if not values:
                return {"min": 0, "max": 0, "avg": 0, "median": 0}
            sorted_vals = sorted(values)
            return {
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "median": sorted_vals[len(sorted_vals) // 2]
            }
        
        analysis = {
            "total_examples": total_examples,
            "tools_per_example": safe_stats(tools_per_example),
            "query_length_words": safe_stats(query_lengths),
            "unique_tools": len(set(tool_names)),
            "taskbench_stats": {
                "seed_range": f"{min(seeds)} - {max(seeds)}" if seeds else "N/A",
                "n_tools_range": f"{min(n_tools_original)} - {max(n_tools_original)}" if n_tools_original else "N/A"
            },
            "distributions": {
                "tools": dict(Counter(tool_names).most_common(20)),
                "task_types": dict(Counter(task_types)),
                "splits": dict(Counter(splits)),
                "configs": dict(Counter(configs))  # Add config distribution
            }
        }
        
        return analysis
    
    def save_dataset(self, mcp_data: List[Dict[str, Any]], analysis: Dict[str, Any]):
        """Save the TaskBench MCP dataset with comprehensive organization"""
        print(f"\n💾 Saving TaskBench MCP dataset...")
        
        if not mcp_data:
            print("❌ No data to save")
            return
        
        # Save main MCP dataset
        mcp_file = self.output_dir / "taskbench_mcp_dataset.json"
        with open(mcp_file, 'w', encoding='utf-8') as f:
            json.dump(mcp_data, f, indent=2, ensure_ascii=False)
        
        # Save analysis
        analysis_file = self.output_dir / "taskbench_mcp_analysis.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        # Save detailed CSV
        csv_data = []
        for example in mcp_data:
            metadata = example.get('metadata', {})
            csv_row = {
                "id": example.get("id", ""),
                "query": example.get("query", "")[:300] + "..." if len(example.get("query", "")) > 300 else example.get("query", ""),
                "query_word_count": len(example.get("query", "").split()),
                "num_tools": len(example.get("tools", [])),
                "tool_names": " | ".join([t.get("name", "") for t in example.get("tools", [])]),
                "domain": metadata.get("domain", ""),
                "difficulty": metadata.get("difficulty", ""),
                "task_type": metadata.get("task_type", ""),
                "multi_step": metadata.get("multi_step", False),
                "split": metadata.get("split", ""),
                "has_solution": "expected_output" in example,
                "task_id": metadata.get("task_id", "")
            }
            csv_data.append(csv_row)
        
        df = pd.DataFrame(csv_data)
        csv_file = self.output_dir / "taskbench_mcp_dataset.csv"
        df.to_csv(csv_file, index=False)
        
        # Save organized subsets
        subsets_created = []
        
        # By difficulty
        difficulty_groups = {}
        for example in mcp_data:
            difficulty = example.get('metadata', {}).get('difficulty', 'medium')
            if difficulty not in difficulty_groups:
                difficulty_groups[difficulty] = []
            difficulty_groups[difficulty].append(example)
        
        for difficulty, examples in difficulty_groups.items():
            if examples:
                subset_file = self.output_dir / f"taskbench_mcp_{difficulty}.json"
                with open(subset_file, 'w', encoding='utf-8') as f:
                    json.dump(examples, f, indent=2, ensure_ascii=False)
                subsets_created.append(f"{difficulty}: {len(examples)} examples")
        
        # By domain
        domain_groups = {}
        for example in mcp_data:
            domain = example.get('metadata', {}).get('domain', 'general')
            if domain not in domain_groups:
                domain_groups[domain] = []
            domain_groups[domain].append(example)
        
        for domain, examples in domain_groups.items():
            if examples and len(examples) >= 5:  # Only save domains with enough examples
                subset_file = self.output_dir / f"taskbench_mcp_{domain.replace(' ', '_')}.json"
                with open(subset_file, 'w', encoding='utf-8') as f:
                    json.dump(examples, f, indent=2, ensure_ascii=False)
        
        # Multi-step vs single-step
        multi_step = [ex for ex in mcp_data if ex.get('metadata', {}).get('multi_step', False)]
        single_step = [ex for ex in mcp_data if not ex.get('metadata', {}).get('multi_step', False)]
        
        if multi_step:
            multi_file = self.output_dir / "taskbench_mcp_multi_step.json"
            with open(multi_file, 'w', encoding='utf-8') as f:
                json.dump(multi_step, f, indent=2, ensure_ascii=False)
        
        if single_step:
            single_file = self.output_dir / "taskbench_mcp_single_step.json"
            with open(single_file, 'w', encoding='utf-8') as f:
                json.dump(single_step, f, indent=2, ensure_ascii=False)
        
        print(f"✅ TaskBench MCP dataset saved:")
        print(f"   📄 Main dataset: {mcp_file}")
        print(f"   📊 Analysis: {analysis_file}")
        print(f"   📈 CSV: {csv_file}")
        print(f"   🎯 Multi-step: {len(multi_step)} examples")
        print(f"   🎯 Single-step: {len(single_step)} examples")
        
        for subset_info in subsets_created:
            print(f"   📁 {subset_info}")
        
        print(f"   📁 Total examples: {len(mcp_data)}")
        
        return mcp_file, analysis_file, csv_file
    
    def print_detailed_analysis(self, analysis: Dict[str, Any]):
        """Print comprehensive analysis of TaskBench dataset"""
        print(f"\n📊 TaskBench MCP Dataset Analysis:")
        print("=" * 60)
        
        print(f"📈 Total Examples: {analysis.get('total_examples', 0)}")
        print(f"🛠️  Unique Tools: {analysis.get('unique_tools', 0)}")
        
        # TaskBench specific stats
        taskbench_stats = analysis.get('taskbench_stats', {})
        print(f"🎲 Seed Range: {taskbench_stats.get('seed_range', 'N/A')}")
        print(f"🔢 Original n_tools Range: {taskbench_stats.get('n_tools_range', 'N/A')}")
        
        # Tools per example
        tools_stats = analysis.get('tools_per_example', {})
        print(f"\n🔧 Tools per Example:")
        print(f"   Min: {tools_stats.get('min', 0)}")
        print(f"   Max: {tools_stats.get('max', 0)}")
        print(f"   Average: {tools_stats.get('avg', 0):.1f}")
        print(f"   Median: {tools_stats.get('median', 0)}")
        
        # Query statistics
        query_stats = analysis.get('query_length_words', {})
        print(f"\n📝 Query Length (words):")
        print(f"   Min: {query_stats.get('min', 0)}")
        print(f"   Max: {query_stats.get('max', 0)}")
        print(f"   Average: {query_stats.get('avg', 0):.1f}")
        
        distributions = analysis.get('distributions', {})
        
        # Most common tools
        tools_dist = distributions.get('tools', {})
        if tools_dist:
            print(f"\n🔧 Most Common Tools:")
            for tool, count in list(tools_dist.items())[:10]:
                print(f"   {tool}: {count}")
        
        # Task types
        types_dist = distributions.get('task_types', {})
        if types_dist:
            print(f"\n🎯 Task Types:")
            for task_type, count in types_dist.items():
                print(f"   {task_type}: {count}")
        
        # Config distribution
        configs_dist = distributions.get('configs', {})
        if configs_dist:
            print(f"\n⚙️  TaskBench Configs:")
            for config, count in configs_dist.items():
                print(f"   {config.title()}: {count}")
    
    def print_sample_examples(self, mcp_data: List[Dict[str, Any]], num_samples: int = 3):
        """Print detailed sample examples from TaskBench"""
        print(f"\n📋 Sample TaskBench MCP Examples (showing {min(num_samples, len(mcp_data))}):")
        print("=" * 80)
        
        for i, example in enumerate(mcp_data[:num_samples]):
            print(f"\n🔹 Example {i+1}:")
            print(f"   ID: {example.get('id', 'N/A')}")
            print(f"   Query: {example.get('query', 'N/A')}")
            
            tools = example.get('tools', [])
            print(f"   Tools ({len(tools)}):")
            
            for j, tool in enumerate(tools):
                print(f"      🛠️  {j+1}. {tool.get('name', 'unnamed')}")
                print(f"         Description: {tool.get('description', 'No description')}")
                
                schema = tool.get('inputSchema', {})
                props = schema.get('properties', {})
                required = schema.get('required', [])
                
                if props:
                    print(f"         Parameters:")
                    for param_name, param_info in props.items():
                        param_type = param_info.get('type', 'unknown')
                        is_required = " (required)" if param_name in required else ""
                        print(f"           - {param_name}: {param_type}{is_required}")
                        if 'description' in param_info:
                            print(f"             {param_info['description']}")
            
            # Metadata
            metadata = example.get('metadata', {})
            print(f"   Metadata:")
            print(f"      Domain: {metadata.get('domain', 'N/A')}")
            print(f"      Difficulty: {metadata.get('difficulty', 'N/A')}")
            print(f"      Task Type: {metadata.get('task_type', 'N/A')}")
            print(f"      Multi-step: {metadata.get('multi_step', False)}")
            print(f"      Split: {metadata.get('split', 'N/A')}")
            
            if 'task_id' in metadata:
                print(f"      Task ID: {metadata['task_id']}")
            
            # Expected output/solution
            if 'expected_output' in example:
                expected = example['expected_output']
                if isinstance(expected, list):
                    print(f"   Solution Steps ({len(expected)}):")
                    for step_num, step in enumerate(expected[:3], 1):  # Show first 3 steps
                        if isinstance(step, dict):
                            action = step.get('action', 'N/A')
                            print(f"      {step_num}. {action}")
                            if 'parameters' in step:
                                params = step['parameters']
                                if isinstance(params, dict) and params:
                                    param_preview = list(params.keys())[:2]  # Show first 2 param names
                                    print(f"         Parameters: {', '.join(param_preview)}")
                    
                    if len(expected) > 3:
                        print(f"      ... and {len(expected) - 3} more steps")
                else:
                    expected_str = str(expected)[:150] + "..." if len(str(expected)) > 150 else str(expected)
                    print(f"   Expected Output: {expected_str}")
            
            if i < num_samples - 1:
                print("-" * 40)

def main():
    """Main function to run Microsoft TaskBench to MCP conversion"""
    print("🚀 Microsoft TaskBench to Model Context Protocol Converter")
    print("=" * 70)
    
    # Initialize converter
    converter = TaskBenchToMCPConverter()
    
    # Download TaskBench dataset
    taskbench_data = converter.download_taskbench_data()
    
    if not taskbench_data:
        print("❌ Failed to download TaskBench data")
        return
    
    # Convert to MCP format
    mcp_data = converter.convert_to_mcp_format(taskbench_data)
    
    if not mcp_data:
        print("❌ Failed to convert to MCP format")
        return
    
    # Analyze dataset
    analysis = converter.analyze_dataset(mcp_data)
    
    # Print detailed analysis
    converter.print_detailed_analysis(analysis)
    
    # Save dataset with comprehensive organization
    files = converter.save_dataset(mcp_data, analysis)
    
    # Show sample examples
    converter.print_sample_examples(mcp_data)
    
    print(f"\n🎉 Microsoft TaskBench to MCP Conversion Complete!")
    print(f"   💾 Files saved in: {converter.output_dir}")
    print(f"   📊 Comprehensive analysis and organization complete!")
    print(f"   🎯 Multi-step and single-step tasks separated!")
    print(f"   🏷️  Domain-specific subsets created!")
    print(f"   ⚡ Difficulty-based organization available!")
    print(f"   🔧 Ready for advanced function calling training!")

if __name__ == "__main__":
    main()