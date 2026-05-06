from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel
from typing import List, Annotated
from dotenv import load_dotenv
import os

load_dotenv()

chat_gemini_api_key = os.getenv('chat_gemini_api_key')

class OutputSchema(BaseModel):
    Reason: Annotated[str, 'High level reasoning behind the issue.'] = None
    how_to_overcome: Annotated[List[str], 'Step by step method on how to overcome the issue.'] = None

chat_gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=chat_gemini_api_key, max_tokens=10000)
structured_chat_gemini = chat_gemini.with_structured_output(OutputSchema)

chat_prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful AI assistant, having over {yoe} years of experience in {role} field.'),
    ('human', '{user_prompt}')
])

chat_prompt_messages = ChatPromptTemplate.from_messages([
    SystemMessage(content='Act as person having expertise in {role} field and has over {yoe} years of experience.'),
    HumanMessage(content='Let me know if you are able to understand the instructions and make sure your answer is in concise form.'),
    AIMessage(content='Yes, able to understood and will answer back in concise mannner'),
    HumanMessage(content='{query}'),
])

yoe = 10
role = 'Cricket Batter Coach'
user_prompt = """
'Whenever I a trying to hit a ball straight down the ground, the ball always hits at the sticker of the bat. You need to provide two exact things. 1. Reasons behind it; 2. How to overcome this. 
"""

print(chat_prompt_messages.invoke({'yoe': yoe, 'role': role, 'user_prompt': user_prompt}))



# formatted_prompt = chat_prompt.invoke({'yoe': yoe, 'role': role, 'user_prompt': user_prompt})
# print('----------------')
# print(formatted_prompt)
# print('----------------')


# # Send prompt
# response = chat_gemini.invoke(formatted_prompt)    
# print('----------------')
# print(response)
# print('----------------')
# response = """
# content='Alright, as an experienced batting coach, I\'ve seen this issue countless times. Hitting the ball on the sticker when trying to go straight down the ground is a clear sign that your bat path and point of contact need some fine-tuning. Let\'s break it down precisely.\n\n---\n\n### 1. Reasons Behind It:\n\nThe ball hitting the sticker (the top part of the bat face, near the splice) indicates that you are making contact with the *top* of the ball, rather than the middle or slightly underneath it. Here are the primary reasons:\n\n*   **Vertical Bat Path / "Chopping Down":** This is the most common culprit. Your bat is coming down too steeply, almost like you\'re chopping wood, rather than sweeping through the line of the ball with a slightly angled, more horizontal arc. This means the bat face is presented vertically at contact, making the sticker the first point of impact.\n*   **Playing Too Late / Too Close to the Body:** If you\'re not getting your front foot out far enough to the pitch of the ball, or if your timing is slightly off, you end up playing the ball *under your eyes* or *too close to your body*. In this position, the bat has already started its upward arc or is forced into a vertical angle, leading to the sticker contact as you try to scoop or push it.\n*   **Head Falling Over / Not Still:** Your head is your guide. If your head falls away to the off-side (for a right-hander) or isn\'t still and directly over the point of impact, it can cause your bat path to become compromised, often leading to that steep, vertical descent onto the ball. You might also be losing sight of the ball just before contact.\n*   **Overly Dominant Bottom Hand:** While the bottom hand provides power, an overactive or overly dominant bottom hand too early in the shot can lead to a \'flicking\' or \'chopping\' motion, preventing the top hand from guiding the bat through a full, clean arc.\n\n---\n\n### 2. How to Overcome This:\n\nTo correct this, we need to adjust your bat path, contact point, and body mechanics. Here are two exact things to focus on:\n\n1.  **Correct Your Bat Path & Point of Contact (The "Brush the Grass" & "Extended Arm" Principle):**\n    *   **Goal:** Ensure your bat comes through a slightly wider, more sweeping arc, meeting the ball *out in front of your front pad* and *just under the middle of the ball*, allowing the sweet spot to hit.\n    *   **Actionable Steps/Drills:**\n        *   **"Brush the Grass" Drill:** When you shadow bat, practice throwdowns, or even during net sessions, consciously aim to **brush the grass** (or ground) with the bottom of your bat *just after* where the ball would have been pitched. This encourages a longer, flatter, more sweeping arc through the hitting zone, preventing the steep, downward chop that leads to sticker contact. Think of it as hitting *through* the ball rather than *down* on it.\n        *   **"Extended Arm" Contact Point:** Visualize and actively try to make contact with the ball when your front arm is **almost fully extended**, well out in front of your body, just outside your front knee. This forces you to get your front foot out to the pitch of the ball, preventing you from playing too close or too late. It ensures you meet the ball at the ideal point in its trajectory, where your bat can present its full face.\n        *   **Drill:** Get a coach or friend to give you soft toss or underarm feeds from 5-7 yards away. Focus exclusively on getting your front foot out, reaching for the ball with an almost extended front arm, and finishing your swing by brushing the ground.\n\n2.  **Master Head Position & Top Hand Control (The "Head on the Coin" & "Top Hand Leads" Method):**\n    *   **Goal:** Keep your head still and directly over the point of impact, guiding the bat with your top hand to present the full face for a clean strike.\n    *   **Actionable Steps/Drills:**\n        *   **"Head on the Coin" Drill:** Before you play the shot, imagine a coin placed flat on top of your head. Your goal is for that coin **not to fall off** throughout the entire shot – from backlift to follow-through. This ensures your head stays still, doesn\'t fall away, and remains directly over the point of impact, which is crucial for maintaining balance and a consistent bat path. Without a stable head, your eyes can\'t track the ball effectively, and your body will compensate with a poor bat path.\n        *   **"Top Hand Leads" Focus:** During shadow batting and throwdowns, consciously focus on your **top hand** (the one closest to the handle) guiding the bat downwards and through the line of the ball. The top hand is responsible for the direction and the initial part of the bat path. The bottom hand\'s role is primarily for power and completion of the shot *after* contact, not to dictate the initial bat path. This helps prevent the \'chopping\' motion often caused by an overactive bottom hand and promotes a smoother, more controlled swing.\n        *   **Drill:** Practice shadow batting in front of a mirror, paying close attention to your head\'s stillness and feeling your top hand initiating and guiding the bat through the downswing.\n\n---\n\nBy diligently focusing on these two areas – correcting your bat path to sweep through the ball out in front, and maintaining a stable head with top-hand guidance – you will significantly reduce sticker contact and start driving the ball cleanly down the ground. Consistency comes from conscious, repeated practice of the correct technique. Good luck!' additional_kwargs={} response_metadata={'finish_reason': 'STOP', 'model_name': 'gemini-2.5-flash', 'safety_ratings': [], 'model_provider': 'google_genai'} id='lc_run--019ce65d-3da8-74a1-b519-8fc3686d0418-0' tool_calls=[] invalid_tool_calls=[] usage_metadata={'input_tokens': 73, 'output_tokens': 3090, 'total_tokens': 3163, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 1856}}
# """


# # Send prompt
# response = structured_chat_gemini.invoke(formatted_prompt)    
# print('----------------')
# print(response)
# print('----------------')
# # Output
# Reason="The primary reasons for hitting the ball on the sticker (top part of the bat) when attempting a straight drive are often an early drop of the lead elbow, leading to a steep bat path, or an incorrect head position that causes you to misjudge the ball's height. Additionally, a lack of positive front foot movement towards the pitch of the ball can force you to reach, resulting in a higher impact point on the bat. Timing issues, where you're slightly late on the ball, can also cause it to ride up the bat face." how_to_overcome=['Keep your head absolutely still and directly over the point of impact. Your eyes must track the ball all the way until it hits the bat.', 'Develop a positive and aggressive front foot stride, aiming to get your front foot as close as possible to the pitch of the ball. This allows you to hit the ball under your eyes and with better balance.', 'Focus on keeping your lead elbow high and driving the bat through a straight, vertical line. This ensures the bat face is perfectly straight at impact and the bat comes down on the ball rather than from above it.', "Maintain soft hands and use your wrists to guide the bat through the line of the ball, ensuring the bat face remains straight and doesn't 'chop' down from a high position.", 'Practice with underarm feeds or off a batting tee, specifically focusing on hitting the sweet spot and driving through the line with a straight bat. This helps engrain the correct bat path and impact point.', 'Perform shadow batting in front of a mirror, concentrating on a high lead elbow, a straight bat path, and maintaining a still head throughout the shot. Visualize hitting the middle of the bat.', 'Place a cone or target a short distance straight down the ground and aim to hit it consistently during practice. This reinforces the correct bat path, follow-through, and encourages hitting through the line.']

