import os
import traceback

def get_leading_spaces(line):
    """Counts the leading whitespace characters."""
    return len(line) - len(line.lstrip(' \t'))

def heal_indentations(directory="."):
    print("=== Commencing Automated Syntax & Indentation Hunt ===")
    
    py_files = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.endswith('.py'):
                py_files.append(os.path.join(root, f))

    fixed_count = 0
    for file_path in py_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()

        attempts = 0
        while attempts < 10:
            try:
                # Try to compile the file into an AST. If it succeeds, no syntax/indent errors exist!
                compile(source, file_path, 'exec')
                break 
            except (IndentationError, TabError) as e:
                line_num = e.lineno
                lines = source.split('\n')
                
                if line_num is None or line_num > len(lines):
                    print(f"[-] Could not map error in {file_path}")
                    break

                offending_line = lines[line_num - 1]
                
                # Look backwards for the closest non-empty line to inherit its indentation
                prev_line_num = line_num - 2
                while prev_line_num >= 0 and not lines[prev_line_num].strip():
                    prev_line_num -= 1
                
                if prev_line_num < 0:
                    break 
                    
                prev_line = lines[prev_line_num]
                target_indent = get_leading_spaces(prev_line)
                
                # If the previous line opened a block (e.g., if, try, for, def), indent +4 spaces
                if prev_line.rstrip().endswith(':'):
                    target_indent += 4
                # If the previous line closed a block/returned, un-indent -4 spaces
                elif any(prev_line.strip().startswith(kw) for kw in ['return', 'continue', 'break']):
                    target_indent = max(0, target_indent - 4)
                    
                # Apply the dynamically calculated indentation to the broken line
                lines[line_num - 1] = (' ' * target_indent) + offending_line.lstrip(' \t')
                source = '\n'.join(lines)
                attempts += 1
                
            except SyntaxError as e:
                print(f"[!] Standard SyntaxError in {file_path} at line {e.lineno}: {e.msg}")
                # Print the broken line so you can easily spot it
                if e.lineno:
                    lines = source.split('\n')
                    print(f"    Code: {lines[e.lineno - 1].strip()}")
                break
            except Exception as e:
                print(f"[!] Unknown compilation error in {file_path}: {e}")
                break
        
        # If we had to make changes, save the healed file
        if attempts > 0:
            try:
                compile(source, file_path, 'exec') # Final sanity check
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(source)
                print(f"[+] Successfully healed IndentationError in {os.path.basename(file_path)} (took {attempts} iteration(s))")
                fixed_count += 1
            except Exception as e:
                print(f"[-] Auto-heal failed for {os.path.basename(file_path)} after {attempts} attempts. Manual review required.")

    print(f"\n=== Done! Healed {fixed_count} files across the repository. ===")

if __name__ == "__main__":
    heal_indentations(".")