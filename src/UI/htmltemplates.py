"""
####################################################################################
#####                 File name: apphtmltemplates.py                           #####
#####                         Author: Ravi Dhir                                #####
#####                      Created on: 12/04/2024                              #####
#####                   HTML and CSS Templates for UI                          #####
####################################################################################
"""
css = '''
<style>
.chat-message {
    padding: 0.8rem;
    border-radius: 5px;  /* Adjusted for slight rounding */
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    font-size: 0.9rem;
}
.chat-message.user {
    background-color: #ffffff;
    justify-content: flex-end;  /* Changed to flex-end for proper alignment */
}
.chat-message.bot {
    background-color: #ffffff;
    justify-content: flex-start;  /* Changed to flex-start for proper alignment */
}
.chat-message .avatar img {
    width: 50px;
    height: 50px;
    border-radius: 50%;  /* Ensures the avatar is circular */
    object-fit: cover;
    margin: 0 10px;
}
.chat-message .message {
    padding: 0.5rem 1rem;
    color: #333;  /* Dark grey color for text, ensuring good readability */
    background-color: #eef2f7;  /* Light blue background for messages */
    border-radius: 15px;  /* Rounded corners for message bubbles */
    max-width: 80%;
    word-wrap: break-word;  /* Ensures text does not overflow */
}
.stButton>button {
    color: white;  /* Changed text color to black for better visibility */
    background-color: #f63366;  /* Bright red color for the button */
    border-radius: 5px;
    border: none;  /* Removed border for a cleaner look */
    padding: 10px 20px;  /* Added padding for better button sizing */
    margin-top: 23px;  /* Ensured margin to align with other elements vertically */
    transition: background-color 0.3s ease;  /* Smooth transition for hover effect */
}
.stButton>button:hover {
    background-color: #cc2a49;  /* Darker shade of the button color on hover */
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://github.com/insculptor/CompleteRAG/blob/main/img/bot.png?raw=true" alt="Bot">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="message">{{MSG}}</div>
    <div class="avatar">
        <img src="https://raw.githubusercontent.com/insculptor/CompleteRAG/main/img/user.png" height="50" alt="User">
    </div>    
</div>
'''
