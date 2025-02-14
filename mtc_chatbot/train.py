import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

# First, let's ensure we have all required NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

# Define a simple tokenizer function
def simple_tokenize(text):
    # Split text into words using spaces and remove punctuation
    words = text.lower().replace('?', ' ').replace('!', ' ').replace('.', ' ').replace(',', ' ').split()
    return words

# Load the intents file
intents = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": [
                "Hi", "Hey", "Hello", "Good morning", "Good afternoon",
                "Good evening", "Is anyone there?", "Hi there"
            ],
            "responses": [
                "Hello! How can I assist you with HR matters today?",
                "Hi there! I'm your HR assistant. What can I help you with?",
                "Hello! I'm here to help with any HR-related questions."
            ]
        },
        {
            "tag": "benefits",
            "patterns": [
                "What benefits do we offer?",
                "Tell me about employee benefits",
                "Health insurance information",
                "What insurance plans are available?",
                "Benefits package details",
                "What's included in our benefits?"
            ],
            "responses": [
                "Our benefits package includes health, dental, and vision insurance, 401(k) matching, paid time off, and life insurance.",
                "We offer comprehensive benefits including medical coverage, retirement plans, and various insurance options.",
                "Employees receive full healthcare coverage, retirement benefits, and additional perks like wellness programs."
            ]
        },
        {
            "tag": "pto",
            "patterns": [
                "How many vacation days do I get?",
                "What's the PTO policy?",
                "How does vacation time work?",
                "Paid time off questions",
                "Holiday policy",
                "When can I take vacation?"
            ],
            "responses": [
                "Full-time employees receive 15 days of PTO annually, plus 10 paid holidays.",
                "Our PTO policy provides 15 vacation days per year, accruing monthly.",
                "You get 15 PTO days annually, with additional time based on tenure."
            ]
        },
        {
            "tag": "salary",
            "patterns": [
                "When do we get paid?",
                "What's the pay schedule?",
                "How often is payroll?",
                "Paycheck questions",
                "Direct deposit information",
                "Payment schedule"
            ],
            "responses": [
                "Employees are paid bi-weekly on Fridays through direct deposit.",
                "Payday is every other Friday via direct deposit.",
                "We process payroll bi-weekly, with payments made on Fridays."
            ]
        },
        {
            "tag": "performance_review",
            "patterns": [
                "When are performance reviews?",
                "How often are evaluations?",
                "Performance evaluation process",
                "Annual review questions",
                "How do reviews work?"
            ],
            "responses": [
                "Performance reviews are conducted annually in December, with mid-year check-ins in June.",
                "We have annual reviews in December and informal quarterly check-ins.",
                "Performance evaluations occur annually with regular feedback sessions throughout the year."
            ]
        },
        {
            "tag": "training",
            "patterns": [
                "What training is available?",
                "Professional development options",
                "Learning opportunities",
                "Training programs",
                "Skill development",
                "Career growth"
            ],
            "responses": [
                "We offer various training programs including online courses, workshops, and certification support.",
                "Employees have access to LinkedIn Learning, internal workshops, and professional development funds.",
                "Training options include technical skills, soft skills, and leadership development programs."
            ]
        },
        {
            "tag": "dress_code",
            "patterns": [
                "What's the dress code?",
                "What should I wear to work?",
                "Dress policy",
                "Office attire guidelines",
                "Casual Friday policy"
            ],
            "responses": [
                "We maintain a business casual dress code, with Casual Fridays allowing jeans.",
                "The office dress code is business casual. Please avoid athletic wear and flip-flops.",
                "Business casual attire is expected, with more casual options allowed on Fridays."
            ]
        },
        {
            "tag": "complaints",
            "patterns": [
                "How do I file a complaint?",
                "Report harassment",
                "Workplace issues",
                "Problem with coworker",
                "Grievance process",
                "Report inappropriate behavior"
            ],
            "responses": [
                "Please report any workplace issues to HR immediately. You can submit confidential complaints through our HR portal or email HR directly.",
                "Contact HR immediately for any workplace concerns. All reports are handled confidentially.",
                "Use our confidential reporting system or speak directly with HR about any workplace issues."
            ]
        },
        {
            "tag": "remote_work",
            "patterns": [
                "Can I work from home?",
                "Remote work policy",
                "Work from home guidelines",
                "Hybrid work options",
                "Remote work equipment",
                "Virtual work policy"
            ],
            "responses": [
                "We offer a hybrid work model with 2-3 days in office per week.",
                "Remote work is available with manager approval. Standard schedule is hybrid.",
                "Our flexible work policy allows for both remote and in-office work arrangements."
            ]
        },
        {
            "tag": "resignation",
            "patterns": [
                "How do I resign?",
                "Resignation process",
                "Two weeks notice",
                "Leaving the company",
                "Notice period",
                "Exit process"
            ],
            "responses": [
                "Please submit a formal resignation letter to your manager and HR with at least two weeks notice.",
                "The resignation process requires written notice and completion of an exit interview.",
                "To resign, provide written notice to your manager and HR, then schedule an exit interview."
            ]
        }
    ]
}

# Save intents to file
with open('intents.json', 'w') as file:
    json.dump(intents, file)

# Initialize lists
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# Process patterns and responses
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Use our simple tokenizer instead of nltk.word_tokenize
        word_list = simple_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

print(f"Unique words: {len(words)}")
print(f"Classes: {len(classes)}")
print(f"Documents: {len(documents)}")

# Save words and classes
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Prepare training data
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Create and train the model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

print("Training the model...")
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5')
print("Model trained and saved")