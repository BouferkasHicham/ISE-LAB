"""
File Tools - Safe file operations within the sandbox directory.

This module provides secure file read/write operations that are restricted
to the designated sandbox directory, preventing agents from modifying
files outside the allowed scope.
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class FileInfo:
    """Information about a file."""
    path: str
    name: str
    extension: str
    size: int
    is_python: bool
    content: Optional[str] = None


class SandboxViolationError(Exception):
    """Raised when an operation attempts to access files outside the sandbox."""
    pass


class FileTools:
    """
    Secure file operations tool for agents.
    
    All operations are restricted to the sandbox directory to ensure
    agents cannot modify system files or files outside the project scope.
    
    Attributes:
        sandbox_root: The root directory where file operations are allowed.
    """
    
    def __init__(self, sandbox_root: str):
        """
        Initialize FileTools with a sandbox directory.
        
        Args:
            sandbox_root: The root directory for all file operations.
        """
        self.sandbox_root = os.path.abspath(sandbox_root)
        self._ensure_sandbox_exists()
    
    def _ensure_sandbox_exists(self) -> None:
        """Create the sandbox directory if it doesn't exist."""
        os.makedirs(self.sandbox_root, exist_ok=True)
    
    def _validate_path(self, filepath: str) -> str:
        """
        Validate that a path is within the sandbox and return the absolute path.
        
        Args:
            filepath: The path to validate.
            
        Returns:
            The absolute, validated path.
            
        Raises:
            SandboxViolationError: If the path is outside the sandbox.
        """
        # Resolve the absolute path
        abs_path = os.path.abspath(os.path.join(self.sandbox_root, filepath))
        
        # If filepath is already absolute, use it directly but validate
        if os.path.isabs(filepath):
            abs_path = os.path.abspath(filepath)
        
        # Security check: ensure the path is within sandbox
        # Use os.path.commonpath for proper path comparison (avoids prefix attacks)
        # e.g., /sandbox vs /sandbox_evil would both start with "/sandbox" using startswith
        sandbox_with_sep = self.sandbox_root.rstrip(os.sep) + os.sep
        abs_path_normalized = abs_path.rstrip(os.sep)
        
        # Check if path is exactly the sandbox or is inside it
        if abs_path_normalized != self.sandbox_root.rstrip(os.sep) and not abs_path.startswith(sandbox_with_sep):
            raise SandboxViolationError(
                f"❌ SECURITY: Access denied! Path '{filepath}' is outside sandbox '{self.sandbox_root}'"
            )
        
        return abs_path
    
    def read_file(self, filepath: str) -> str:
        """
        Read the contents of a file within the sandbox.
        
        Args:
            filepath: Path to the file (relative to sandbox or absolute).
            
        Returns:
            The file contents as a string.
            
        Raises:
            SandboxViolationError: If the path is outside the sandbox.
            FileNotFoundError: If the file doesn't exist.
        """
        abs_path = self._validate_path(filepath)
        
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"File not found: {abs_path}")
        
        with open(abs_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def write_file(self, filepath: str, content: str) -> str:
        """
        Write content to a file within the sandbox.
        
        Creates parent directories if they don't exist.
        
        Args:
            filepath: Path to the file (relative to sandbox or absolute).
            content: The content to write.
            
        Returns:
            The absolute path of the written file.
            
        Raises:
            SandboxViolationError: If the path is outside the sandbox.
        """
        abs_path = self._validate_path(filepath)
        
        # Create parent directories if needed
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        
        with open(abs_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return abs_path
    
    def backup_file(self, filepath: str) -> str:
        """
        Create a backup of a file before modification.
        
        Args:
            filepath: Path to the file to backup.
            
        Returns:
            Path to the backup file.
        """
        abs_path = self._validate_path(filepath)
        
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Cannot backup non-existent file: {abs_path}")
        
        backup_path = abs_path + ".backup"
        shutil.copy2(abs_path, backup_path)
        
        return backup_path
    
    def restore_backup(self, filepath: str) -> bool:
        """
        Restore a file from its backup.
        
        Args:
            filepath: Path to the original file.
            
        Returns:
            True if restored successfully, False if no backup exists.
        """
        abs_path = self._validate_path(filepath)
        backup_path = abs_path + ".backup"
        
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, abs_path)
            os.remove(backup_path)
            return True
        
        return False
    
    def cleanup_backups(self) -> int:
        """
        Remove all .backup files from the sandbox.
        
        Returns:
            Number of backup files removed.
        """
        count = 0
        for root, dirs, files in os.walk(self.sandbox_root):
            for filename in files:
                if filename.endswith(".backup"):
                    backup_path = os.path.join(root, filename)
                    os.remove(backup_path)
                    count += 1
        return count
    
    def list_python_files(self, subdir: str = "") -> List[str]:
        """
        List all Python files in the sandbox or a subdirectory.
        
        Args:
            subdir: Optional subdirectory to search within.
            
        Returns:
            List of Python file paths relative to the sandbox.
        """
        search_path = self._validate_path(subdir) if subdir else self.sandbox_root
        python_files = []
        
        for root, _, files in os.walk(search_path):
            for file in files:
                if file.endswith('.py'):
                    full_path = os.path.join(root, file)
                    # Return path relative to sandbox root
                    rel_path = os.path.relpath(full_path, self.sandbox_root)
                    python_files.append(rel_path)
        
        return sorted(python_files)

    def list_files(self, subdir: str = "") -> List[str]:
        """
        List all filenames in the sandbox or a subdirectory.
        
        Args:
            subdir: Optional subdirectory to search within.
            
        Returns:
            List of filenames in the directory.
        """
        search_path = self._validate_path(subdir) if subdir else self.sandbox_root
        if not os.path.exists(search_path) or not os.path.isdir(search_path):
            return []
            
        return os.listdir(search_path)
    
    def get_file_info(self, filepath: str) -> FileInfo:
        """
        Get detailed information about a file.
        
        Args:
            filepath: Path to the file.
            
        Returns:
            FileInfo object with file details.
        """
        abs_path = self._validate_path(filepath)
        
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"File not found: {abs_path}")
        
        path_obj = Path(abs_path)
        
        return FileInfo(
            path=filepath,
            name=path_obj.name,
            extension=path_obj.suffix,
            size=os.path.getsize(abs_path),
            is_python=path_obj.suffix == '.py',
            content=None  # Load content separately if needed
        )
    
    def get_all_files_info(self) -> List[FileInfo]:
        """
        Get information about all Python files in the sandbox.
        
        Returns:
            List of FileInfo objects.
        """
        python_files = self.list_python_files()
        return [self.get_file_info(f) for f in python_files]
    
    def file_exists(self, filepath: str) -> bool:
        """
        Check if a file exists within the sandbox.
        
        Args:
            filepath: Path to the file.
            
        Returns:
            True if the file exists, False otherwise.
        """
        try:
            abs_path = self._validate_path(filepath)
            return os.path.exists(abs_path)
        except SandboxViolationError:
            return False
    
    def delete_file(self, filepath: str) -> bool:
        """
        Delete a file within the sandbox.
        
        Args:
            filepath: Path to the file.
            
        Returns:
            True if deleted, False if file didn't exist.
        """
        abs_path = self._validate_path(filepath)
        
        if os.path.exists(abs_path):
            os.remove(abs_path)
            return True
        
        return False
    
    def get_directory_structure(self) -> Dict[str, Any]:
        """
        Get the directory structure of the sandbox as a nested dictionary.
        
        Returns:
            Dictionary representing the directory structure.
        """
        structure = {"name": os.path.basename(self.sandbox_root), "type": "directory", "children": []}
        
        def build_tree(path: str, node: Dict[str, Any]) -> None:
            try:
                entries = sorted(os.listdir(path))
            except PermissionError:
                return
            
            for entry in entries:
                full_path = os.path.join(path, entry)
                if os.path.isdir(full_path):
                    child = {"name": entry, "type": "directory", "children": []}
                    build_tree(full_path, child)
                    node["children"].append(child)
                else:
                    node["children"].append({
                        "name": entry,
                        "type": "file",
                        "extension": Path(entry).suffix
                    })
        
        build_tree(self.sandbox_root, structure)
        return structure
    
    def copy_directory(self, source: str, destination: str) -> str:
        """
        Copy a directory within or into the sandbox.
        
        Args:
            source: Source directory path.
            destination: Destination path within sandbox.
            
        Returns:
            Absolute path of the copied directory.
        """
        dest_path = self._validate_path(destination)
        
        # Source can be outside sandbox (for initial setup)
        source_path = os.path.abspath(source)
        
        if os.path.exists(dest_path):
            shutil.rmtree(dest_path)
        
        shutil.copytree(source_path, dest_path)
        return dest_path
