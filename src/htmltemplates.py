css = '''
<style>
.chat-message {
    padding: 0.8rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    font-size: 0.9rem;
    border: 1px solid #e1e4e8;
}
.chat-message.user {
    background-color: #f9f9f9;
    justify-content: end;
}
.chat-message.bot {
    background-color: #ffffff;
    justify-content: start;
}
.chat-message .avatar img {
    width: 50px;
    height: 50px;
    border-radius: 50%;
    object-fit: cover;
    margin: 0 10px;
}
.chat-message .message {
    padding: 0.5rem 1rem;
    color: #333;
    background-color: #eef2f7;
    border-radius: 15px;
    max-width: 80%;
    word-wrap: break-word;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png" alt="Bot">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="message">{{MSG}}</div>
    <div class="avatar">
        <img src="https://i.ibb.co/rdZC7LZ/Photo-logo-1.png" alt="User">
    </div>    
</div>
'''