#!/usr/bin/env python3
# Add send_message alias to TelegramUltra
import sys
import os

# Read telegram_ultra.py
telegram_file = 'integrations/telegram_ultra.py'

try:
    with open(telegram_file, 'r') as f:
        content = f.read()
    
    # Check if send_message already exists
    if 'def send_message' in content and 'send_text' in content:
        print('✅ send_message method already exists')
        sys.exit(0)
    
    # Find the class definition and add alias
    if 'class TelegramUltra' in content:
        # Add alias after __init__
        insert_point = content.find('def __init__')
        if insert_point > 0:
            # Find end of __init__ method
            next_def = content.find('\n    def ', insert_point + 10)
            if next_def > 0:
                alias_code = '''\n    async def send_message(self, message: str):
        """Alias for send_text - compatibility"""
        return await self.send_text(message)
'''
                content = content[:next_def] + alias_code + content[next_def:]
                
                with open(telegram_file, 'w') as f:
                    f.write(content)
                
                print('✅ Added send_message alias to TelegramUltra')
            else:
                print('⚠️  Could not find insertion point')
        else:
            print('⚠️  __init__ not found')
    else:
        print('⚠️  TelegramUltra class not found')

except FileNotFoundError:
    print('⚠️  telegram_ultra.py not found')
except Exception as e:
    print(f'❌ Error: {e}')
